import warnings
import random
import numpy as np
import os
import pandas as pd

import paddle
import paddle.distributed as dist
from .SampleMethod import theSM

from paddle import Model
from paddle import fluid
from paddle.hapi.model import DynamicGraphAdapter
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.executor import global_scope
from paddle.fluid.framework import in_dygraph_mode, Variable
from paddle.fluid.layers import collective

from paddleslim.nas.ofa import OFA
from paddle.fluid.layers.utils import flatten
from paddle.hapi.callbacks import config_callbacks, EarlyStopping
from paddle.io import Dataset, DistributedBatchSampler, DataLoader

from ..dataset.dataiter import NewDataLoader as TrainDataLoader
from ..dataset.random_size_crop import MyRandomResizedCrop


def _all_gather(x, nranks, ring_id=0, use_calc_stream=True):
    return collective._c_allgather(
        x, nranks, ring_id=ring_id, use_calc_stream=use_calc_stream)


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def to_numpy(var):
    assert isinstance(var, (Variable, fluid.core.VarBase)), "not a variable"
    if isinstance(var, fluid.core.VarBase):
        return var.numpy()
    t = global_scope().find_var(var.name).get_tensor()
    return np.array(t)


def _update_input_info(inputs):
    "Get input shape list by given inputs in Model initialization."
    shapes = None
    dtypes = None
    if isinstance(inputs, list):
        shapes = [list(input.shape) for input in inputs]
        dtypes = [input.dtype for input in inputs]
    elif isinstance(inputs, dict):
        shapes = [list(inputs[name].shape) for name in inputs]
        dtypes = [inputs[name].dtype for name in inputs]
    else:
        return None
    return shapes, dtypes

def up2eight(ratio, channel_num):
    cn = ratio * channel_num
    if cn % 8 == 0:
        return cn
    return cn + 8 - cn % 8

def near2eight(ratio, channel_num, divisor=8, min_val=None):
    v = ratio * channel_num
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def arch2size(arch=None):
    ratio        = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    # base_channel = [64, 128, 256, 512]
    base_channel = [64, 64*1.1, 64*1.7, 64*1.9]
    channel_dict = {i: x for i, x in enumerate(ratio, 1)}
    stem_ratio   = channel_dict[int(arch[5])]
    stage1ratio  = [channel_dict[int(ss)] for ss in arch[6:16].replace('0', '')]
    stage2ratio  = [channel_dict[int(ss)] for ss in arch[16:26].replace('0', '')]
    stage3ratio  = [channel_dict[int(ss)] for ss in arch[26:42].replace('0', '')]
    stage4ratio  = [channel_dict[int(ss)] for ss in arch[42:].replace('0', '')]
    total_ratios = [stage1ratio, stage2ratio, stage3ratio, stage4ratio]
    # total_channel_num = []
    total_channel_num = [up2eight(stem_ratio, 64)]
    for i, llist in enumerate(total_ratios):
        base_c = base_channel[i]
        for val in llist:
            total_channel_num.append(up2eight(val, base_c))
    return sum(total_channel_num)

class MyDynamicGraphAdapter(DynamicGraphAdapter):
    def __init__(self, model, cfg=None):
        self.model = model
        self._nranks = ParallelEnv().nranks
        self._local_rank = ParallelEnv().local_rank
        self._merge_count = {
            'eval_total': 0,
            'test_total': 0,
            'eval_batch': 0,
            'test_batch': 0
        }

        self._input_info = None
        if self._nranks > 1:
            dist.init_parallel_env()
            if isinstance(self.model.network, OFA):
                self.model.network.model = paddle.DataParallel(self.model.network.model, find_unused_parameters=True)
                self.ddp_model = self.model.network
            else:
                self.ddp_model = paddle.DataParallel(self.model.network)
        self.dyna_bs           = cfg.get('dynamic_batch_size', 1)
        self.sample_method     = cfg.get('sample_method')
        self.use_loss_expand   = cfg.get('use_loss_expand')
        self.bias_sample_flag  = cfg.get('bias_sample_flag')
        if self.bias_sample_flag:
            self.stage1_sample_num = cfg.get('stage1_sample_num')
            self.sub_net_sample = theSM(method="random")
            # print('[DEBUG] stage1_sample_num: {}'.format(self.stage1_sample_num))

    # TODO multi device in dygraph mode not implemented at present time
    def train_batch(self, inputs, labels=None, **kwargs):
        assert self.model._optimizer, "model not ready, please call `model.prepare()` first"
        # self.model.network.train()
        self.model.network.model.train()
        self.mode        = 'train'
        inputs           = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels           = labels or []
        labels           = [to_variable(l) for l in to_list(labels)]
        epoch            = kwargs.get('epoch', None)
        self.epoch       = epoch
        nBatch           = kwargs.get('nBatch', None)
        step             = kwargs.get('step', None)
        max_arch         = '1558511111111111111111111111111111111111111111111111'
        max_arch_size    = arch2size(max_arch)
        if self.bias_sample_flag:
            stage1sampel = {}
            for i in range(self.stage1_sample_num):
                subnet_seed = int('%d%.1d' % (epoch * nBatch + step, i))
                np.random.seed(subnet_seed)
                self.model.network.active_subnet(img_size=MyRandomResizedCrop.current_size, sample_fun=self.sub_net_sample)
                net_arch  = self.model.network.gen_subnet_code
                arch_size = arch2size(net_arch)
                if len(stage1sampel) < self.dyna_bs:
                    stage1sampel[arch_size] = net_arch
                else:
                    if arch_size > min(stage1sampel.keys()):
                        del stage1sampel[min(stage1sampel.keys())]
                        stage1sampel[arch_size] = net_arch
            if step % 3 ==0 :
                while len(stage1sampel) < self.dyna_bs+1:
                    random_seed =  int('%d%.1d' % (epoch * nBatch + step, self.stage1_sample_num))
                    np.random.seed(subnet_seed)
                    self.model.network.active_subnet(img_size=MyRandomResizedCrop.current_size, sample_fun=self.sub_net_sample)
                    net_arch  = self.model.network.gen_subnet_code
                    arch_size = arch2size(net_arch) 
                    stage1sampel[arch_size] = net_arch
 
            # print('[DEBUG] sample num: {}'.format(len(stage1sampel)))
                
            for arch_size, net_arch in stage1sampel.items():
                self.model.network.active_subnet(arch=net_arch)
                if self._nranks > 1:
                    outputs = self.ddp_model.forward(* [to_variable(x) for x in inputs])
                else:
                    outputs = self.model.network.forward(* [to_variable(x) for x in inputs])
                losses       = self.model._loss(*(to_list(outputs) + labels))
                losses       = to_list(losses)
                final_loss   = fluid.layers.sum(losses)
                if self.use_loss_expand:
                    net_arch     = self.model.network.gen_subnet_code
                    expand_ratio = arch2size(net_arch) / max_arch_size
                    final_loss  *= expand_ratio
                final_loss.backward()
        else:
            for i in range(self.dyna_bs):
                subnet_seed = int('%d%.1d' % (epoch * nBatch + step, i))
                np.random.seed(subnet_seed)
                self.model.network.active_subnet(img_size=MyRandomResizedCrop.current_size, sample_fun=self.sub_net_sample)
                if self._nranks > 1:
                    outputs = self.ddp_model.forward(* [to_variable(x) for x in inputs])
                else:
                    outputs = self.model.network.forward(* [to_variable(x) for x in inputs])

                losses       = self.model._loss(*(to_list(outputs) + labels))
                losses       = to_list(losses)
                final_loss   = fluid.layers.sum(losses)
                if self.use_loss_expand:
                    net_arch     = self.model.network.gen_subnet_code
                    expand_ratio = arch2size(net_arch) / max_arch_size
                    final_loss  *= expand_ratio
                final_loss.backward()

        self.model._optimizer.step()
        self.model._optimizer.clear_grad()

        metrics = []
        for metric in self.model._metrics:
            metric_outs = metric.compute(*(to_list(outputs) + labels))
            m = metric.update(* [to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        return ([to_numpy(l) for l in losses], metrics) if len(metrics) > 0 else [to_numpy(l) for l in losses]

    def eval_batch(self, inputs, labels=None):
        self.model.network.eval()
        self.model.network.model.eval()
        self.mode = 'eval'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = labels or []
        labels = [to_variable(l) for l in to_list(labels)]

        outputs = self.model.network.forward(* [to_variable(x) for x in inputs])
        if self.model._loss:
            losses = self.model._loss(*(to_list(outputs) + labels))
            losses = to_list(losses)

        if self._nranks > 1:
            outputs = [_all_gather(o, self._nranks) for o in to_list(outputs)]
            labels = [_all_gather(l, self._nranks) for l in labels]
        metrics = []
        for metric in self.model._metrics:
            # cut off padding value.
            if self.model._test_dataloader is not None and self._nranks > 1 \
                    and isinstance(self.model._test_dataloader, DataLoader):
                total_size = len(self.model._test_dataloader.dataset)
                samples = outputs[0].shape[0]
                current_count = self._merge_count.get(self.mode + '_total', 0)
                if current_count + samples >= total_size:
                    outputs = [
                        o[:int(total_size - current_count)] for o in outputs
                    ]
                    labels = [
                        l[:int(total_size - current_count)] for l in labels
                    ]
                    self._merge_count[self.mode + '_total'] = 0
                    self._merge_count[self.mode + '_batch'] = int(total_size - current_count)
                else:
                    self._merge_count[self.mode + '_total'] += samples
                    self._merge_count[self.mode + '_batch'] = samples

            metric_outs = metric.compute(*(to_list(outputs) + labels))
            m = metric.update(* [to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        if self.model._loss and len(metrics):
            return [to_numpy(l) for l in losses], metrics
        elif self.model._loss:
            return [to_numpy(l) for l in losses]
        else:
            return metrics
    
    def save(self, path):
        params = self.model.network.state_dict()
        fluid.save_dygraph(params, path)
        if self.model._optimizer is None:
            return
        if self.model._optimizer.state_dict():
            optim = self.model._optimizer.state_dict()
            optim['epoch'] = self.epoch
            fluid.save_dygraph(optim, path)


class Trainer(Model):
    def __init__(self, network, inputs=None, labels=None, cfg=None):
        # super().__init__(network, inputs=inputs, labels=labels)

        self.mode = 'train'
        self.network = network
        self._inputs = None
        self._labels = None
        self._loss = None
        self._loss_weights = None
        self._optimizer = None
        self._input_info = None
        self._is_shape_inferred = False
        self._test_dataloader = None
        self.stop_training = False

        self._inputs = self._verify_spec(inputs, is_input=True)
        self._labels = self._verify_spec(labels)
        # init backend
        self._adapter = MyDynamicGraphAdapter(self, cfg)
        self.start_epoch = 0
    
    def fit(
            self,
            train_data=None,
            eval_data=None,
            batch_size=1,
            epochs=1,
            eval_freq=1,
            log_freq=10,
            save_dir=None,
            save_freq=1,
            verbose=2,
            drop_last=False,
            shuffle=True,
            num_workers=0,
            callbacks=None, ):
        assert train_data is not None, "train_data must be given!"

        if isinstance(train_data, Dataset):
            train_sampler = DistributedBatchSampler(
                train_data,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
            train_loader = TrainDataLoader(
                train_data,
                batch_sampler=train_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
        else:
            train_loader = train_data

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(
                eval_data, batch_size=batch_size)
            eval_loader = DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=0,
                return_list=True)
        elif eval_data is not None:
            eval_loader = eval_data
        else:
            eval_loader = None

        do_eval = eval_loader is not None
        self._test_dataloader = eval_loader

        steps = self._len_data_loader(train_loader)
        cbks = config_callbacks(
            callbacks,
            model=self,
            epochs=epochs,
            steps=steps,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self._metrics_name(), )

        if any(isinstance(k, EarlyStopping) for k in cbks) and not do_eval:
            warnings.warn("EarlyStopping needs validation data.")

        cbks.on_begin('train')
        gt_ls =[]

        for epoch in range(self.start_epoch, epochs):
            cbks.on_epoch_begin(epoch)
            logs = self._run_one_epoch(train_loader, cbks, 'train', epoch=epoch)
            cbks.on_epoch_end(epoch, logs)

            if do_eval and epoch % eval_freq == 0:

                eval_steps = self._len_data_loader(eval_loader)
                cbks.on_begin('eval', {
                    'steps': eval_steps,
                    'metrics': self._metrics_name()
                })

                eval_logs = self._run_one_epoch(eval_loader, cbks, 'eval')

                cbks.on_end('eval', eval_logs)
                if self.stop_training:
                    break

        cbks.on_end('train', logs)
        self._test_dataloader = None

    def train_batch(self, inputs, labels=None, **kwargs):
        loss = self._adapter.train_batch(inputs, labels, **kwargs)
        if fluid.in_dygraph_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    def _run_one_epoch(self, data_loader, callbacks, mode, logs={}, **kwargs):    
        outputs = []
        if mode == 'train':
            MyRandomResizedCrop.epoch = kwargs.get('epoch', None)

        for step, data in enumerate(data_loader):
            # data might come from different types of data_loader and have
            # different format, as following:
            # 1. DataLoader in static graph:
            #    [[input1, input2, ..., label1, lable2, ...]]
            # 2. DataLoader in dygraph
            #    [input1, input2, ..., label1, lable2, ...]
            # 3. custumed iterator yield concated inputs and labels:
            #   [input1, input2, ..., label1, lable2, ...]
            # 4. custumed iterator yield seperated inputs and labels:
            #   ([input1, input2, ...], [label1, lable2, ...])
            # To handle all of these, flatten (nested) list to list.
            data = flatten(data)
            # LoDTensor.shape is callable, where LoDTensor comes from
            # DataLoader in static graph

            batch_size = data[0].shape()[0] if callable(data[0].shape) else data[0].shape[0]

            callbacks.on_batch_begin(mode, step, logs)

            if mode != 'predict':
                if mode == 'train':
                    MyRandomResizedCrop.sample_image_size(step)
                    outs = getattr(self, mode + '_batch')(data[:len(self._inputs)],
                                                          data[len(self._inputs):], 
                                                          epoch=kwargs.get('epoch', None),
                                                          nBatch=len(data_loader),
                                                          step=step)
                else:
                    outs = getattr(self, mode + '_batch')(data[:len(self._inputs)], data[len(self._inputs):])
                if self._metrics and self._loss:
                    metrics = [[l[0] for l in outs[0]]]
                elif self._loss:
                    metrics = [[l[0] for l in outs]]
                else:
                    metrics = []

                # metrics
                for metric in self._metrics:
                    res = metric.accumulate()
                    metrics.extend(to_list(res))

                assert len(self._metrics_name()) == len(metrics)
                for k, v in zip(self._metrics_name(), metrics):
                    logs[k] = v
            else:
                if self._inputs is not None:
                    outs = self.predict_batch(data[:len(self._inputs)])
                else:
                    outs = self.predict_batch(data)

                outputs.append(outs)

            logs['step'] = step
            if mode == 'train' or self._adapter._merge_count.get(mode + '_batch', 0) <= 0:
                logs['batch_size'] = batch_size * ParallelEnv().nranks
            else:
                logs['batch_size'] = self._adapter._merge_count[mode + '_batch']

            callbacks.on_batch_end(mode, step, logs)
        self._reset_metrics()

        if mode == 'predict':
            return logs, outputs
        return logs

    def evaluate(
            self,
            eval_data,
            batch_size=1,
            log_freq=10,
            verbose=1,
            eval_sample_num=10,
            num_workers=0,
            callbacks=None):

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(eval_data, batch_size=batch_size)
            eval_loader = paddle.io.DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True, use_shared_memory=True)
        else:
            eval_loader = eval_data

        self._test_dataloader = eval_loader

        cbks = config_callbacks(
            callbacks,
            model=self,
            log_freq=log_freq,
            verbose=verbose,
            metrics=self._metrics_name(), )

        eval_steps = self._len_data_loader(eval_loader)

        self.network.model.eval()
        
        import time
        sample_result = []
        if eval_sample_num != 0:
            eval_num = eval_sample_num
        else:
            arch_path      = os.getcwd()
            modelarch_file = open(arch_path+'/model_str.txt', 'r', encoding='utf-8')
            modelarch_list = modelarch_file.read().strip().split(',')
            eval_num       = len(modelarch_list)
        
        sample_fun = theSM(method="random")
        for i in range(eval_num):
            cbks.on_begin('eval', {'steps': eval_steps, 'metrics': self._metrics_name()})
            # subnet_seed = int(time.time()/10)
            # random.seed(subnet_seed)
            if eval_sample_num != 0:
                self.network.active_subnet(224, sample_fun=sample_fun)
            else:
                self.network.active_subnet(arch=modelarch_list[i])
            logs = self._run_one_epoch(eval_loader, cbks, 'eval')

            cbks.on_end('eval', logs)

            # self._test_dataloader = None

            eval_result = {}
            for k in self._metrics_name():
                eval_result[k] = logs[k]
            sample_res = '{} {} {}'.format(
                self.network.gen_subnet_code, eval_result['acc_top1'], eval_result['acc_top5'])
            if ParallelEnv().local_rank == 0:
                print(sample_res)
            sample_result.append(sample_res)
            if ParallelEnv().local_rank == 0:
                with open('channel_sample.txt', 'a') as f:
                    f.write('{}\n'.format(sample_res))

        return sample_result
        
        
