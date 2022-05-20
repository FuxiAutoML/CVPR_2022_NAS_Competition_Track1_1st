import warnings
import random
import numpy as np
import pandas as pd
import os

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
from collections import defaultdict
from pprint import pprint
from functools import cmp_to_key
import time
# from hnas.models.pinas import ContrastiveHead

TRAIN_BREAK = -1
EVAL_BREAK = -1

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

def Arch2Ratio(arch=None):
    ratio        = [1.0, 0.95, 0.9, 0.85, 0.75, 0.7]
    channel_dict = {i: x for i, x in enumerate(ratio, 1)}
    max_arch_p   = 1 + 5 + 5 + 8 + 5
    stem_ratio   = channel_dict[int(arch[5])]
    blocks_ratio = [channel_dict[int(ss)] for ss in arch[6:].replace('0', '')]
    arch_p       = stem_ratio + sum(blocks_ratio)
    return arch_p / max_arch_p

class MyDynamicGraphAdapter(DynamicGraphAdapter):
    def __init__(self, model, cfg=None,
    pinas_stage=0, main_loss_lambda=0.75, frozen_backbone=True, largest_net_in_stage1=False):
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
        self.dyna_bs        = cfg.get('dynamic_batch_size', 1)
        self.sample_method  = str(cfg.get('sample_method'))
        self.sub_net_sample = theSM(method=self.sample_method)

        self.pinas_stage = pinas_stage
        # self.pinas_loss_head = ContrastiveHead(temperature=0.2)
        self.main_loss_lambda = main_loss_lambda
        self.frozen_backbone = frozen_backbone
        self.largest_net_in_stage1 = largest_net_in_stage1
        print('[DEBUG]self.frozen_backbone', self.frozen_backbone)
        if self.largest_net_in_stage1:
            if self._nranks > 1:
                self.ddp_model.set_largest_subnet()
            else:
                self.model.network.set_largest_subnet()

    # TODO multi device in dygraph mode not implemented at present time
    def train_batch(self, inputs, labels=None, **kwargs):
        assert self.model._optimizer, "model not ready, please call `model.prepare()` first"
        # self.model.network.train()
        self.model.network.model.train()
        self.mode = 'train'
        inputs = to_list(inputs)
        self._input_info = _update_input_info(inputs)
        labels = labels or []
        labels = [to_variable(l) for l in to_list(labels)]
        epoch = kwargs.get('epoch', None)
        self.epoch = epoch
        nBatch = kwargs.get('nBatch', None)
        step = kwargs.get('step', None)

        if self.pinas_stage >= 1:
            # for i in range(self.dyna_bs):
            #     subnet_seed = int('%d%.1d' % (epoch * nBatch + step, i))
            #     np.random.seed(subnet_seed)
            #     # if i == 0:      # ! 训练最大网路
            #     #     arch = '1558511111111111111111111111111111111111111111111111'
            #     #     self.model.network.active_subnet(arch=arch, set_depth_list=True)
            #     # else:           # ! 采用抽样策略抽样
            #     self.model.network.active_subnet(
            #         img_size=MyRandomResizedCrop.current_size,
            #         sample_fun=self.sub_net_sampler, split=False
            #     )
            #     arch_code = self.model.network.gen_subnet_code
            #     self.model.network.active_subnet(arch=arch_code, set_depth_list=True, split=self.split, split_num=self.split_num)
            #     # print('[DEBUG]arch_code@train_batch', arch_code)
            #     assert self.split == False, "PiNAS training shalle NOT be splitted for now"
            #     if self._nranks > 1:
            #         outputs = self.ddp_model.pinas_forward_classifier(* [to_variable(x) for x in inputs], frozen_backbone=self.frozen_backbone, both_teacher=False)
            #     else:
            #         outputs = self.model.network.pinas_forward_classifier(* [to_variable(x) for x in inputs], frozen_backbone=self.frozen_backbone, both_teacher=False)
            #         # running on local GPU machine will go this way
            #     # print('[DEBUG]outputs', outputs)
            #     # print('[DEBUG]labels', labels)
            #     # print('[DEBUG]outputs')
            #     # pprint(outputs)
            #     # print('[DEBUG]labels')
            #     # pprint(labels)
            #     losses = self.model._loss(*(to_list(outputs) + labels))
            #     losses = to_list(losses) # ! seems like just a sum of teacher and student, exactly two loss tensor here
            #     # print('[DEBUG]losses', losses) # hard classification loss and teacher distill loss
            #     if self.main_loss_lambda == 1:
            #         final_loss = losses[0]
            #     elif self.main_loss_lambda > 0:
            #         final_loss = (losses[0] * self.main_loss_lambda + losses[1] * (1.0 - self.main_loss_lambda)) * 2.0
            #     else:
            #         # final_loss = fluid.layers.sum([losses[0] * self.main_loss_lambda, losses[1] * (1.0 - self.main_loss_lambda)]) * 2.0
            #         final_loss = fluid.layers.sum(losses)
            #     final_loss.backward()
            for i in range(self.dyna_bs):
                subnet_seed = int('%d%.1d' % (epoch * nBatch + step, i))
                np.random.seed(subnet_seed)
                # if i == 0:      # 训练最大网路
                #     arch = '1558511111111111111111111111111111111111111111111111'
                #     self.model.network.active_subnet(arch=arch)
                # if i == 1:      # 训练最小网络
                #     arch = '1222233333000000323200000043530000000000006414000000'
                #     self.model.network.active_subnet(arch=arch)
                # else:           # 采用抽样策略抽样
                self.model.network.active_subnet(img_size=MyRandomResizedCrop.current_size, sample_fun=self.sub_net_sample)
                # print(self.model.network.gen_subnet_code)
                if self._nranks > 1:
                    outputs = self.ddp_model.pinas_forward_classifier(* [to_variable(x) for x in inputs], 
                    frozen_backbone=self.frozen_backbone, both_teacher=False)
                else:
                    outputs = self.model.network.pinas_forward_classifier(* [to_variable(x) for x in inputs],
                    frozen_backbone=self.frozen_backbone, both_teacher=False)

                losses       = self.model._loss(*(to_list(outputs) + labels))
                losses       = to_list(losses)
                final_loss   = fluid.layers.sum(losses)
                net_arch     = self.model.network.gen_subnet_code
                # expand_ratio = Arch2Ratio(net_arch)
                # final_loss  *= expand_ratio
                final_loss.backward()

            self.model._optimizer.step()
            self.model._optimizer.clear_grad()

            metrics = []
            for metric in self.model._metrics:
                metric_outs = metric.compute(*(to_list(outputs) + labels))
                m = metric.update(* [to_numpy(m) for m in to_list(metric_outs)])
                metrics.append(m)

        return ([to_numpy(l) for l in losses], metrics) if len(metrics) > 0 else [to_numpy(l) for l in losses]

    def train_batch2(self, inputs, inputs2, labels=None, **kwargs):
        assert self.model._optimizer, "model not ready, please call `model.prepare()` first"
        # self.model.network.train()
        self.model.network.model.train()
        self.mode = 'train'
        inputs = to_list(inputs) # inputs is X, which is B,C,H,W
        inputs2 = to_list(inputs2)
        self._input_info = _update_input_info(inputs)
        labels = labels or []
        labels = [to_variable(l) for l in to_list(labels)]
        epoch = kwargs.get('epoch', None)
        self.epoch = epoch
        nBatch = kwargs.get('nBatch', None)
        step = kwargs.get('step', None)
        
        if self.pinas_stage == 0:
            paths = []
            if self.largest_net_in_stage1:
                pass
            else:
                for i in range(2):
                    subnet_seed = int('%d%.1d' % (epoch * nBatch + step, i))
                    np.random.seed(subnet_seed)
                    # ! 采用抽样策略抽样
                    self.model.network.active_subnet(
                        img_size=MyRandomResizedCrop.current_size,
                        sample_fun=self.sub_net_sample
                    )
                    paths.append(
                        self.model.network.gen_subnet_code
                    )


            if self._nranks > 1:
                loss_generated0, k0 = self.ddp_model.pinas_forward_step_by_step2(
                    * [to_variable(x) for x in inputs], * [to_variable(x) for x in inputs2], paths=paths, _nranks=self._nranks, batch_ddp=True, use_einsum=True
                )
                loss_generated0.backward()
                loss_generated1, k1 = self.ddp_model.pinas_forward_step_by_step2(
                    * [to_variable(x) for x in inputs], * [to_variable(x) for x in inputs2], paths=paths[::-1], _nranks=self._nranks, batch_ddp=True, use_einsum=True
                )
                loss_generated1.backward()
                # loss_0and1 = loss_generated0 + loss_generated1
                # loss_0and1.backward()
                with paddle.no_grad():
                    self.ddp_model._dequeue_and_enqueue((k0 + k1) / 2, _nranks=self._nranks)
            else:
                loss_generated0, k0 = self.model.network.pinas_forward_step_by_step2(
                    * [to_variable(x) for x in inputs], * [to_variable(x) for x in inputs2], paths=paths, use_einsum=True
                )
                    # print('[DEBUG]loss_generated0', loss_generated0)
                loss_generated0.backward()
                loss_generated1, k1 = self.model.network.pinas_forward_step_by_step2(
                    * [to_variable(x) for x in inputs], * [to_variable(x) for x in inputs2], paths=paths[::-1], use_einsum=True
                )
                loss_generated1.backward()
                # loss_0and1 = loss_generated0 + loss_generated1
                # loss_0and1.backward()
                    # print('[DEBUG]loss_generated1', loss_generated1)
                with paddle.no_grad():
                    self.model.network._dequeue_and_enqueue((k0 + k1) / 2, _nranks=self._nranks)

            losses = to_list([loss_generated0, loss_generated1])
            self.model._optimizer.step()
            self.model._optimizer.clear_grad()
            metrics = []

        else:
            raise NotImplementedError()

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
                    self._merge_count[self.mode + '_batch'] = int(total_size -
                                                                  current_count)
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
    def __init__(self, network, inputs=None, labels=None, cfg=None,
    pinas_stage=0, student_loss_lambda=-1, frozen_backbone=True, largest_net_in_stage1=False):
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
        self._adapter = MyDynamicGraphAdapter(self, cfg,
        pinas_stage=pinas_stage, main_loss_lambda=student_loss_lambda,
        frozen_backbone=frozen_backbone ,largest_net_in_stage1=largest_net_in_stage1)
        self.start_epoch = 0
    
    def fit(
            self,
            train_data=None,
            eval_data=None,
            train_data2=None,
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
            train_loader = paddle.io.DataLoader(
                train_data,
                batch_sampler=train_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
            if train_data2 is not None:
                train_loader2 = paddle.io.DataLoader(
                    train_data2,
                    batch_sampler=train_sampler,
                    places=self._place,
                    num_workers=num_workers,
                    return_list=True)
        else:
            train_loader = train_data

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(
                eval_data, batch_size=batch_size)
            eval_loader = paddle.io.DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=0,
                return_list=True, use_shared_memory=True)
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
        if do_eval:
            arch_path  = os.getcwd()
            gt_file = open(arch_path+'/cur_ground_truth', 'r', encoding='utf-8')
            gt_ls_full = gt_file.read().strip().split('\n')
            gt_arch = [x.split("\t")[0] for x in gt_ls_full]  
            gt_score1 = [float(x.split("\t")[1]) for x in gt_ls_full]  
            gt_score5 = [float(x.split("\t")[2]) for x in gt_ls_full]  

        for epoch in range(self.start_epoch, epochs):
            cbks.on_epoch_begin(epoch)
            if train_data2 is None:
                logs = self._run_one_epoch(train_loader, cbks, 'train', epoch=epoch)
            else:
                logs = self._run_one_epoch_2(train_loader, train_loader2, cbks, 'train', epoch=epoch)
            cbks.on_epoch_end(epoch, logs)

            if do_eval and epoch % eval_freq == 0:

                eval_steps = self._len_data_loader(eval_loader)
          
                self.network.model.eval()     #what?????????
                sample_result = [] 
                #arch_path      = os.getcwd()
                #modelarch_file = open(arch_path+'/model_str.txt', 'r', encoding='utf-8')
                #modelarch_list = modelarch_file.read().strip().split(',')
                eval_num       = len(gt_arch)
      
                corr_ls1 = [] 
                corr_ls5 = [] 
                for i in range(eval_num): 
                    cbks.on_begin('eval', {'steps': eval_steps, 'metrics': self._metrics_name()})
                    self.network.active_subnet(arch=gt_arch[i])
                    eval_logs = self._run_one_epoch(eval_loader, cbks, 'eval')
          
                    eval_result = {}
                    for k in self._metrics_name():
                        eval_result[k] = logs[k]
                    sample_res = '{} {} {}'.format(
                        self.network.gen_subnet_code, eval_result['acc_top1'], eval_result['acc_top5'])
                    corr_ls1.append(eval_result['acc_top1'])
                    corr_ls5.append(eval_result['acc_top5'])
                    if ParallelEnv().local_rank == 0:
                        print(sample_res)
                    #sample_result.append(sample_res)
                    #if ParallelEnv().local_rank == 0:
                    #    with open('channel_samplett.txt', 'a') as f:
                    #         f.write('{}\n'.format(sample_res))
                    cbks.on_end('eval', eval_logs)
               
                s1 = pd.Series(gt_score1)
                s2 = pd.Series(gt_score5)
                s11 = pd.Series(corr_ls1)
                s22 = pd.Series(corr_ls5)
                kendall1 = s1.corr(s11,method="kendall")
                pearson1 = s1.corr(s11,method="pearson")
                kendall5 = s2.corr(s22,method="kendall")
                pearson5 = s2.corr(s22,method="pearson")
                if ParallelEnv().local_rank == 0:
                    print("gt_corr_eval-------------start")
                    print("kendall_top1: ",kendall1)
                    print("pearson_top1: ",pearson1)
                    print("kendall_top5: ",kendall5)
                    print("pearson_top5: ",pearson5)
                    print("gt_corr_eval--------------end")
                #cbks.on_begin('eval', {
                #    'steps': eval_steps,
                #    'metrics': self._metrics_name()
                #})

                #eval_logs = self._run_one_epoch(eval_loader, cbks, 'eval')

                if self.stop_training:
                    break

        cbks.on_end('train', logs)
        self._test_dataloader = None

    def train_batch(self, inputs, labels=None, **kwargs):
        loss = self._adapter.train_batch(inputs, labels, **kwargs)
        if fluid.in_dygraph_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    def train_batch2(self, inputs, inputs2, labels=None, **kwargs):
        # ! the most important loss producer function
        loss = self._adapter.train_batch2(inputs, inputs2, labels, **kwargs)
        if fluid.in_dygraph_mode() and self._input_info is None:
            self._update_inputs()
        return loss

    def _run_one_epoch(self, data_loader, callbacks, mode, logs={}, **kwargs):    
        outputs = []
        if mode == 'train':
            MyRandomResizedCrop.epoch = kwargs.get('epoch', None)

        for step, data in enumerate(data_loader):
            if mode=='train':
                if TRAIN_BREAK > 0 and step > TRAIN_BREAK:
                    break
            if mode=='eval':
                if EVAL_BREAK > 0 and step > EVAL_BREAK: 
                    break
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
                if self._metrics and self._loss and isinstance(outs, tuple):
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

    def _run_one_epoch_2(self, data_loader, data_loader2, callbacks, mode, logs={}, **kwargs):    
        # ! let's admit that this is just a log producer
        assert mode=='train', "mode is not train, shall not call _run_one_epoch_2!"
        outputs = []
        if mode == 'train':
            MyRandomResizedCrop.epoch = kwargs.get('epoch', None)

        for step, (data, data2) in enumerate(zip(data_loader, data_loader2)):
            if mode=='train':
                if TRAIN_BREAK > 0 and step > TRAIN_BREAK:
                    break

            data = flatten(data)
            data2 = flatten(data2)
            batch_size = data[0].shape()[0] if callable(data[0].shape) else data[0].shape[0]

            callbacks.on_batch_begin(mode, step, logs)

            if mode != 'predict': # ! mode is eval in evaluate
                if mode == 'train':
                    MyRandomResizedCrop.sample_image_size(step)
                    outs = getattr(self, mode + '_batch2')(data[:len(self._inputs)],
                                                          data2[:len(self._inputs)],
                                                          data[len(self._inputs):], 
                                                          epoch=kwargs.get('epoch', None),
                                                          nBatch=len(data_loader),
                                                          step=step)
                else:
                    raise NotImplementedError()
                if self._metrics and self._loss and isinstance(outs, tuple):
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
        

