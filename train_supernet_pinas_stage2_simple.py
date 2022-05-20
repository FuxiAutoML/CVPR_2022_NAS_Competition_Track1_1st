import os
from pprint import pprint

import paddle
import paddle.nn as nn
from paddle.nn import CrossEntropyLoss
from paddle.vision.transforms import (
    RandomHorizontalFlip, RandomResizedCrop, SaturationTransform, 
    Compose, Resize, HueTransform, BrightnessTransform, ContrastTransform, 
    RandomCrop, Normalize, RandomRotation, CenterCrop)
from paddle.io import DataLoader
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, LinearWarmup

from hnas.utils.callbacks import LRSchedulerM, MyModelCheckpoint
from hnas.utils.transforms import ToArray
from hnas.dataset.random_size_crop import MyRandomResizedCrop
from paddle.vision.datasets import DatasetFolder

from paddleslim.nas.ofa.convert_super import Convert, supernet
from paddleslim.nas.ofa import RunConfig, DistillConfig, ResOFA
from paddleslim.nas.ofa.utils import utils

import paddle.distributed as dist
from hnas.utils.yacs import CfgNode
from hnas.models.builder import build_classifier
from hnas.utils.hapi_wrapper import Trainer
from hnas.models.utils import transfer_original_supernet_params_to_pinas, load_pretrained_to_pinas, set_model_fc_as_inited

import warnings
warnings.simplefilter("ignore", UserWarning)

pinas_setting = {
    "negative_sample_num": -1,
}

def _loss_forward(self, input, tea_input, label=None):
    if label is not None:
        ret = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            name=self.name)

        mse = paddle.nn.functional.cross_entropy(
            input,
            paddle.nn.functional.softmax(tea_input),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=True,
            axis=self.axis)
        # mse = paddle.nn.functional.mse_loss(input, tea_input)
        return ret, mse
    else:
        ret = paddle.nn.functional.cross_entropy(
            input,
            tea_input,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            name=self.name)
        return ret
CrossEntropyLoss.forward = _loss_forward

def _compute(self, pred, tea_pred, label=None, *args):
    if label is None:
        label = tea_pred
    pred = paddle.argsort(pred, descending=True)
    pred = paddle.slice(
        pred, axes=[len(pred.shape) - 1], starts=[0], ends=[self.maxk])
    if (len(label.shape) == 1) or \
        (len(label.shape) == 2 and label.shape[-1] == 1):
        # In static mode, the real label data shape may be different
        # from shape defined by paddle.static.InputSpec in model
        # building, reshape to the right shape.
        label = paddle.reshape(label, (-1, 1))
    elif label.shape[-1] != 1:
        # one-hot label
        label = paddle.argmax(label, axis=-1, keepdim=True)
    correct = pred == label
    return paddle.cast(correct, dtype='float32')

paddle.metric.Accuracy.compute = _compute


def run(
    backbone=       'resnet48',
    image_size=     '224',
    max_epoch=      120,
    lr=             0.0025,
    weight_decay=   3e-5,
    momentum=       0.9,
    batch_size=     80,
    dyna_batch_size=4,
    warmup=         2,
    phase=          None,
    resume=         None,
    pretrained=     'ILSVRC2012/ckpt/resnet48.pdparams',
    image_dir=      'ILSVRC2012/',
    save_dir=       'cvpr_baseline/CVPR_2022_Track1_demo/ckpt/',
    log_freq=       10,
    save_dir_new=   'ckpt/pn',
    save_freq=      5,
    pinas_stage=    1,
    num_workers=    8,
    use_tensor_dataset=False,
    **kwargs
    ):
    run_config = locals()
    run_config.update(run_config["kwargs"])
    del run_config["kwargs"]
    config = CfgNode(run_config)
    config.image_size_list = [int(x) for x in config.image_size.split(',')]

    nprocs = len(paddle.get_cuda_rng_state())
    gpu_str = []
    for x in range(nprocs):
        gpu_str.append(str(x))
    gpu_str = ','.join(gpu_str)
    print(f'gpu num: {nprocs}')
    dist.spawn(main, args=(config,), nprocs=nprocs, gpus=gpu_str)

def main(cfg):
    paddle.set_device('gpu:{}'.format(dist.ParallelEnv().device_id))
    if dist.get_rank() == 0:
        print(cfg)
    IMAGE_MEAN = (0.485,0.456,0.406)
    IMAGE_STD = (0.229,0.224,0.225)

    cfg.lr = cfg.lr * cfg.batch_size * dist.get_world_size() / 256
    warmup_step = int(1281024 / (cfg.batch_size * dist.get_world_size())) * cfg.warmup

    transforms = Compose([
        MyRandomResizedCrop(cfg.image_size_list),
        RandomHorizontalFlip(),
        ToArray(),
        Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    val_transforms = Compose([Resize(256), CenterCrop(224), ToArray(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    train_set = DatasetFolder(os.path.join(cfg.image_dir, 'train'), transform=transforms)
    use_tensor_dataset = cfg.use_tensor_dataset

    val_set = DatasetFolder(os.path.join(cfg.image_dir,'val_slim'), transform=val_transforms)
    callbacks = [LRSchedulerM(), 
                 MyModelCheckpoint(cfg.save_freq, cfg.save_dir, cfg.resume, cfg.phase)]

    net = build_classifier(cfg.backbone, pretrained=cfg.pretrained, reorder=True)
    #net = build_classifier(cg.backbone, reorder=True)
    tnet = build_classifier(cfg.backbone, pretrained=cfg.pretrained, reorder=False)
    #tnet = build_classifier(cfg.backbone, reorder=False)
    origin_weights = {}
    for name, param in net.named_parameters():
        origin_weights[name] = param
    
    sp_model = Convert(supernet(expand_ratio=[1.0])).convert(net)  # net转换成supernet
    utils.set_state_dict(sp_model, origin_weights)  # 重新对supernet加载数据
    del origin_weights

    cand_cfg = {
            'i': [224],  # image size
            'd': [(2, 5), (2, 5), (2, 8), (2, 5)],  # depth
            'k': [3],  # kernel size
            'c': [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7] # channel ratio
    }
    ofa_net = ResOFA(sp_model,  
                     distill_config=DistillConfig(teacher_model=tnet), 
                     candidate_config=cand_cfg,
                     block_conv_num=2)
    ofa_net.set_task('expand_ratio')

    run_config = {
        'dynamic_batch_size': cfg.dyna_batch_size, 
        'sample_method': cfg.sample_method, 
        'use_loss_expand': cfg.use_loss_expand,
        'bias_sample_flag': cfg.bias_sample_flag,
        'stage1_sample_num': cfg.stage1_sample_num
    }
    model = Trainer(ofa_net, cfg=run_config)
    
    use_ckpt = cfg.use_ckpt
    if use_ckpt:
        ckpt_data = paddle.load(cfg.ckpt_dir)
        print('[DEBUG]load ckpt from last stage')
        # pprint(list(ckpt_data.keys()))
        ckpt_data_cleaned = {k:v for k, v in ckpt_data.items() if "ofa_teacher_model" not in k}
        ckpt_data_cleaned = {k:v for k, v in ckpt_data.items() if "queue" not in k}
        ckpt_data_cleaned = {k:v for k, v in ckpt_data.items() if "neck" not in k}
        pprint(list(ckpt_data_cleaned.keys()))
        model.network.set_state_dict(ckpt_data_cleaned)

    model.prepare(
        paddle.optimizer.Momentum(
            learning_rate=LinearWarmup(
                CosineAnnealingDecay(cfg.lr, cfg.max_epoch), warmup_step, 0., cfg.lr),
            momentum=cfg.momentum,
            parameters=model.parameters(),
            weight_decay=cfg.weight_decay),
        CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1,5)))
    train_flag = cfg.train_flag
    if train_flag:
        model.fit(
            train_set,
            None,
            epochs=cfg.max_epoch,
            batch_size=cfg.batch_size,
            save_dir=cfg.save_dir,
            save_freq=cfg.save_freq,
            log_freq=cfg.log_freq,
            shuffle=True,
            num_workers=cfg.num_workers,
            verbose=2, 
            drop_last=True,
            callbacks=callbacks,
        )
    
    #model.evaluate(val_set, batch_size=cfg.batch_size, num_workers=4, eval_sample_num=10)
    if cfg.pinas_stage == 2:
        num_workers = 0 if not use_tensor_dataset else 0
        model.evaluate(val_set, batch_size=cfg.batch_size, num_workers=num_workers, eval_sample_num=0)

if __name__ == '__main__':
    import fire
    fire.Fire({"run": run})
