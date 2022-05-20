import paddle
import paddle.nn as nn
from pprint import pprint
import numpy as np


def adjust_bn_according_to_idx(bn, idx):
    bn_weights = bn.parameters()[0]
    bn_bias = bn.parameters()[1]
    bn_weights.set_value(paddle.index_select(bn_weights, idx, 0))
    bn_bias.set_value(paddle.index_select(bn_bias, idx, 0))

    if type(bn) in [nn.BatchNorm1D, nn.BatchNorm2D]:
        bn_mean = bn.parameters()[2]
        bn_var = bn.parameters()[3]
        bn_mean.set_value(paddle.index_select(bn_mean, idx, 0))
        bn_var.set_value(paddle.index_select(bn_var, idx, 0))


def make_divisible(v, divisor, min_val=None):
    """This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def load_checkpoint_to_model(model, ckpt_path):
    if 'pdparams' not in ckpt_path:
        ckpt_path = ckpt_path + '.pdparams'
    ckpt_data = paddle.load(ckpt_path)
    if 'state_dict' in ckpt_data:
        state_dict = ckpt_data['state_dict']
    else:
        state_dict = ckpt_data
    # print('[DEBUG]ckpt_data')
    # pprint(ckpt_data)
    model.network.set_state_dict(state_dict)
    return ckpt_data

def transfer_original_supernet_params_to_pinas(model, ckpt_data, copy_fc=True):
    print('[DEBUG]transfer_original_supernet_params_to_pinas called')
    # pprint(list(ckpt_data.keys()))
    # print('*'*20)
    try:
        named_params = model.network.named_parameters()
    except:
        named_params = model.named_parameters()
    else:
        pass
    named_params = list(named_params)
    ckpt_data_cleaned = {k.replace('_layers.', ''):v for k, v in ckpt_data.items() if 'ofa_teacher_model' not in k}
    print('[DEBUG]ckpt_data_cleaned:')
    pprint(list(ckpt_data_cleaned.keys()))
    print('[DEBUG]named_params of model:')
    pprint([k for k,v in named_params])
    # print('*'*20)
    model.set_state_dict(ckpt_data_cleaned)
    for name, param in named_params:
        if "ofa_teacher_model" in name:
            continue
        # print(name)
        if 'original_head' in name and copy_fc:
            suffix = name.split('original_head.')[-1]
            original_name = 'model.blocks.24.' + suffix
            print('[DEBUG]original_name', original_name)
            if original_name in ckpt_data:
                print('set {} into {}'.format(original_name, name))
                param.set_value(ckpt_data[original_name])
    # print('-'*20)
    # print('[DEBUG]show ckpt and model diff')
    # pprint(ckpt_data_cleaned.items())
    try:
        named_params = dict(model.network.named_parameters())
    except:
        named_params = dict(model.named_parameters())
    for k, v in ckpt_data_cleaned.items():
        if k in named_params:
            print(k)
            print(paddle.all(named_params[k] == v))

def load_pretrained_to_pinas(pretrained_ckpt, net, copy_fc=True):
    print('[DEBUG] load pretrained to pinas called')
    print('[DEBUG] load from: ', pretrained_ckpt)
    ckpt_data = paddle.load(pretrained_ckpt)
    net.set_state_dict(ckpt_data)
    print('[DEBUG]ckpt data keys')
    pprint(list(ckpt_data.keys()))
    named_params = net.named_parameters()

    for name, param in named_params:
        if 'original_head' in name and copy_fc:
            suffix = name.split('original_head.')[-1]
            original_name = 'blocks.24.' + suffix
            if original_name in ckpt_data:
                print('set {} into {}'.format(original_name, name))
                param.set_value(ckpt_data[original_name])

    return True

def copy_student_init_params_to_teacher(net, tnet):
    net_params = net.named_parameters()
    tnet_params = tnet.named_parameters()
    for param_tnet, param_net in zip(tnet_params, net_params):
        assert param_net[0] == param_tnet[0], "param name inconsistent:{} vs {}".format(param_tnet[0], param_net[0])
        param_tnet[1].set_value(param_net[1])

def set_model_fc_as_inited(trainer, fc_in_size=512, num_classes=1000, method='normal'):
    if method == 'normal':
        # [fc_in_size, num_classes]
        tensor_weights = paddle.normal(mean=0., std=0.01, shape=[fc_in_size, num_classes])
        # [num_classes]
        tensor_bias = paddle.zeros(shape=[num_classes,])
    elif method == 'pytorch':
        K = 1.0 / fc_in_size
        range_ = np.sqrt(K)
        tensor_weights = paddle.uniform(shape=[fc_in_size, num_classes], min=-1.0 * range_, max = range_)
        tensor_bias = paddle.uniform(shape=[num_classes, ], min=-1.0 * range_, max = range_)
    else:
        raise NotImplementedError()

    for k, v in trainer.network.named_parameters():
        if 'teacher' in k:
            continue
        if 'blocks.24.fc.fn.weight' in k:
            # print(k, v)
            v.set_value(tensor_weights)
        elif 'blocks.24.fc.fn.bias' in k:
            # print(k, v)
            v.set_value(tensor_bias)
    
    return

def check_model_fc_as_inited(trainer):
    print('[DEBUG]check_model_fc_as_inited called')
    for k, v in trainer.network.named_parameters():
        if 'teacher' in k:
            continue
        if 'blocks.24.fc.fn.weight' in k:
            print(k, v)
        elif 'blocks.24.fc.fn.bias' in k:
            print(k, v)



