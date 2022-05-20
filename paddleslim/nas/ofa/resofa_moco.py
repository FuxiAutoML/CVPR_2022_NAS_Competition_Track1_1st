import random
import logging
import numpy as np

from collections import OrderedDict
import paddle.nn as nn
from paddle import DataParallel
import paddle
from .pinas_utils import MOCO

from .ofa import OFA
from .layers_base import BaseBlock
from ...core import GraphWrapper, dygraph2program
from .get_sub_model import get_prune_params_config, prune_params, check_search_space
from ...common import get_logger
import functools
from pprint import pprint


_logger = get_logger(__name__, level=logging.INFO)

class ResOFAMOCO(OFA, MOCO): # ! please note that this is asscociated with ResNet class in hnas/models/resnet.py by "backbone" arg
    def __init__(self,
                 model,
                 run_config=None,
                 distill_config=None,
                 elastic_order=None,
                 train_full=False,
                 candidate_config=None,
                 block_conv_num=2,
                 pinas=True,
                 temp=0.2
                 ):
        super().__init__(model, run_config, distill_config, elastic_order, train_full)
        self.model.eval()
        self._clear_search_space()
        self.cand_cfg = candidate_config
        # self.cand_cfg = {
        #     'i': [224],  # image size
        #     'd': [(2, 5), (2, 5), (2, 8), (2, 5)],  # depth
        #     'k': [3],  # kernel size
        #     'c': [1.0, 0.95, 0.9, 0.85, 0.75, 0.7]  # channel ratio
        # }
        self.im_size_dict = {x: i for i, x in enumerate(self.cand_cfg['i'], 1)}
        self.depth_dict = {k: k for s, e in self.cand_cfg['d'] for k in range(s, e+1)}
        self.kernel_dict = {x: i for i, x in enumerate(self.cand_cfg['k'], 1)}
        self.channel_dict = {x: i for i, x in enumerate(self.cand_cfg['c'], 1)}

        assert block_conv_num in [2, 3]
        self.block_conv_num = block_conv_num
        self.grouped_block_index = self.model.grouped_block_index
        
        # ! for pinas
        self.pinas = pinas
        assert self.pinas == True, "pinas shall be True"
        if self.pinas:
            self.pinas_lambda = 0.999
            self.head = ContrastiveHead(temperature=temp)

    def active_subnet(self, img_size=224, arch=None, set_depth_list=False, sample_fun=None):
        """
        arch : str, the sub model architecture in str like: '1558511111111111111111111111111111111111111111111111'
        """
        # print('2222', self._ofa_layers)
        if arch:
            # print(arch)
            im_size_dict = {i: x for i, x in enumerate(self.cand_cfg['i'], 1)}
            channel_dict = {i: x for i, x in enumerate(self.cand_cfg['c'], 1)}
            im_size_code = arch[0]
            depth_code   = arch[1:5]
            stem_code    = arch[5]
            blocks_code  = arch[6:]

            stem_ratio   = channel_dict[int(stem_code)]
            blocks_ratio = [channel_dict[int(x)] for x in blocks_code.replace('0', '')]

            self.act_im_size    = im_size_dict[int(im_size_code)]
            self.act_depth_list = [int(x) for x in depth_code]
            self.current_config = OrderedDict()
            self.current_config['blocks.0.conv'] = {'expand_ratio': stem_ratio}
            for stage_list, d in zip(self.grouped_block_index, self.act_depth_list):
                for i, idx in enumerate(stage_list):
                    if i<d:
                        self.current_config[f'blocks.{idx}.conv1'] = {'expand_ratio': blocks_ratio.pop(0)}
                        self.current_config[f'blocks.{idx}.conv2'] = {'expand_ratio': blocks_ratio.pop(0)}
            for key, v in self._ofa_layers.items():
                if key not in self.current_config:
                    self.current_config[key] = v
        else:
            self.act_im_size    = img_size
            self.act_depth_list = sample_fun.depth_list()
            self.current_config = sample_fun.current_config()
        self.current_config[f'neck.mlp.0'] = {'expand_ratio':1.0}
        self.current_config[f'neck.mlp.2'] = {'expand_ratio':1.0}
        # self.current_config[f'blocks.24.fc'] = {'expand_ratio':1.0}
        #print(self.act_depth_list) 
        #print(self.current_config)
        self._broadcast_ss()
        
        if set_depth_list:
            self.model.act_depth_list = self.act_depth_list

    @property
    def gen_subnet_code(self):
        submodel_code = [self.im_size_dict[self.act_im_size]]
        submodel_code += [d for d in self.act_depth_list]
        submodel_code_str = ''.join([str(x) for x in submodel_code])
        # k_code = ['', '', '', '', '']
        v = self.current_config['blocks.0.conv']
        conv_code = str(self.channel_dict[v['expand_ratio']])
        for stage_list, d in zip(self.grouped_block_index, self.act_depth_list):
            for i, idx in enumerate(stage_list):
                if i < d:
                    v = self.current_config[f'blocks.{idx}.conv1']
                    conv_code += str(self.channel_dict[v['expand_ratio']])
                    v = self.current_config[f'blocks.{idx}.conv2']
                    conv_code += str(self.channel_dict[v['expand_ratio']])
                else:
                    conv_code += '0' * self.block_conv_num
        
        submodel_code_str += conv_code
        
        return submodel_code_str


    def _clear_search_space(self):
        """ find shortcut in model, and clear up the search space """
        self.model.eval()
        _st_prog = dygraph2program(self.model, inputs=[2, 3, 224, 224], dtypes=[np.float32])
        self._same_ss = check_search_space(GraphWrapper(_st_prog))

        self._same_ss = sorted(self._same_ss)
        self._param2key = {}
        self._broadcast = True

        self.universe = []
        ### the name of sublayer is the key in search space
        ### param.name is the name in self._same_ss
        model_to_traverse = self.model._layers if isinstance(self.model, DataParallel) else self.model
        for name, sublayer in model_to_traverse.named_sublayers():
            if isinstance(sublayer, BaseBlock):
                for param in sublayer.parameters():
                    if self._find_ele(param.name, self._same_ss):
                        self._param2key[param.name] = name
                    if 'conv' in name:
                        self.universe.append(name)
        def func(x, y):
            x = x.split('.')
            xk1, xk2 = int(x[1]), x[2]
            y = y.split('.')
            yk1, yk2 = int(y[1]), y[2]
            if xk1 > yk1:
                return 1
            elif xk1 < yk1:
                return -1
            else:
                if xk2 > yk2:
                    return 1
                elif xk2 < yk2:
                    return -1
                else:
                    return 0
        self.universe.sort(key=functools.cmp_to_key(func))
        ### double clear same search space to avoid outputs weights in same ss.
        tmp_same_ss = []
        for ss in self._same_ss:
            per_ss = []
            for key in ss:
                if key not in self._param2key.keys():
                    continue

                if self._param2key[key] in self._ofa_layers.keys() and (
                    'expand_ratio' in self._ofa_layers[self._param2key[key]] or \
                        'channel' in self._ofa_layers[self._param2key[key]]):
                    per_ss.append(key)
                else:
                    _logger.info("{} not in ss".format(key))
            if len(per_ss) != 0:
                tmp_same_ss.append(per_ss)
        self._same_ss = tmp_same_ss

        for per_ss in self._same_ss:
            for ss in per_ss[1:]:
                if 'expand_ratio' in self._ofa_layers[self._param2key[ss]]:
                    self._ofa_layers[self._param2key[ss]].pop('expand_ratio')
                elif 'channel' in self._ofa_layers[self._param2key[ss]]:
                    self._ofa_layers[self._param2key[ss]].pop('channel')
                if len(self._ofa_layers[self._param2key[ss]]) == 0:
                    self._ofa_layers.pop(self._param2key[ss])

    def forward(self, x):
        teacher_output = None
        if self._add_teacher:
            self._reset_hook_before_forward()
            with paddle.no_grad():
                teacher_output = self.ofa_teacher_model.model.forward(x)

        # self.active_subnet()
        # print(self.gen_subnet_code())
        model = self.model._layers if isinstance(self.model, DataParallel) else self.model
        model.act_depth_list = self.act_depth_list
        stu_out = self.model.forward(x)

        if teacher_output is not None and self.training:
            return stu_out, teacher_output
        else:
            return stu_out

    # ! for pinas
    def pinas_forward_classifier(self, x, frozen_backbone=False, both_teacher=False):
        teacher_output = None
        if self._add_teacher:
            self._reset_hook_before_forward()
            with paddle.no_grad():
                # teacher_output = self.ofa_teacher_model.model.forward_backbone(x)
                # teacher_output = self.ofa_teacher_model.model.forward_cls_head(teacher_output)
                teacher_output = self.ofa_teacher_model.model.forward(x)

        # self.active_subnet()
        # print(self.gen_subnet_code())
        model = self.model._layers if isinstance(self.model, DataParallel) else self.model
        model.act_depth_list = self.act_depth_list
        
        if frozen_backbone:
            with paddle.no_grad():
                stu_out = self.model.forward_backbone(x)
            stu_out = self.model.forward_cls_head(stu_out)
        else:
            stu_out = self.model.forward(x)
        
        if teacher_output is not None and self.training:
            if both_teacher:
                return teacher_output, teacher_output
            return stu_out, teacher_output
        else:
            return stu_out

    @paddle.no_grad()
    def update_mean_teacher(self,):
        student_net_params = self.model.parameters()
        teacher_net_params = self.ofa_teacher_model.parameters()

        for i, (t_param, s_param) in enumerate(zip(teacher_net_params, student_net_params)):
            # print(paddle.all(t_param == s_param))
            teacher_net_params[i].set_value(
                t_param * self.pinas_lambda + s_param * (1.0 - self.pinas_lambda)
            )
        # print('[DEBUG]update_mean_teacher done')

    def set_largest_subnet(self, set_for_teacher_also=True, split=False, split_num=-1):
        print('[DEBUG]set largest subnet called')
        arch_code = '1558511111111111111111111111111111111111111111111111'
        self.active_subnet(arch=arch_code, set_depth_list=True)
        if set_for_teacher_also:
            self.ofa_teacher_model.model.active_subnet(arch=arch_code, set_depth_list=True)
        return
    
    def pinas_forward_step_by_step(self, x, paths=[], use_einsum=False, batch_ddp=False, _nranks=1, split=False, split_num=2):
        # q for student
        # k for teacher
        # print('[DEBUG]paths', paths)
        im_q, im_k = x, x

        self._reset_hook_before_forward() 
        with paddle.no_grad():
            if len(paths) == 2:
                self.ofa_teacher_model.model.active_subnet(arch=paths[1], set_depth_list=True)
            self.update_mean_teacher()
            if _nranks > 1 and batch_ddp:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.ofa_teacher_model.model.model.forward_backbone(im_k)
            k = self.ofa_teacher_model.model.model.forward_neck(k) # k: [B, f_dim=128]
            k = nn.functional.normalize(k, axis=1)
            if _nranks > 1  and batch_ddp:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                
        if len(paths) == 2:
            self.active_subnet(arch=paths[0], set_depth_list=True)
        self_model = self.model._layers if isinstance(self.model, DataParallel) else self.model
        q = self_model.forward_backbone(im_q)
        q = self_model.forward_neck(q) # q: [B, f_dim=128]
        q = nn.functional.normalize(q, axis=1)

        if use_einsum:
            l_pos = paddle.einsum('nc,nc->n', *[q, k]).unsqueeze(-1)
            l_neg = paddle.einsum('nc,ck->nk', *[q, self.queue.clone().detach()])
        else:
            l_pos = paddle.sum(paddle.multiply(q, k), axis=-1, keepdim=True)
            l_neg = paddle.matmul(q, self.queue.clone().detach())
        
        loss_generated = self.head(l_pos, l_neg, loss_dict=False)

        # for now we will not use queue mechanism
        # self._dequeue_and_enqueue((k_0 + k_1) / 2)

        return loss_generated, k.clone().detach()
    
    def pinas_forward_step_by_step2(self, x, x2, paths=[], use_einsum=False, batch_ddp=False, _nranks=1, split=False, split_num=2):
        # q for student
        # k for teacher
        # print('[DEBUG]paths', paths)
        im_q, im_k = x, x2
        # print('[DEBUG]im_q', im_q)
        # print('[DEBUG]im_k', im_k)
        self._reset_hook_before_forward() 
        with paddle.no_grad():
            if len(paths) == 2:
                self.ofa_teacher_model.model.active_subnet(arch=paths[1], set_depth_list=True)
            self.update_mean_teacher()
            if _nranks > 1 and batch_ddp:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.ofa_teacher_model.model.model.forward_backbone(im_k)
            k = self.ofa_teacher_model.model.model.forward_neck(k) # k: [B, f_dim=128]
            k = nn.functional.normalize(k, axis=1)
            if _nranks > 1  and batch_ddp:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                
        if len(paths) == 2:
            self.active_subnet(arch=paths[0], set_depth_list=True)
        self_model = self.model._layers if isinstance(self.model, DataParallel) else self.model
        q = self_model.forward_backbone(im_q)
        q = self_model.forward_neck(q) # q: [B, f_dim=128]
        q = nn.functional.normalize(q, axis=1)

        if use_einsum:
            l_pos = paddle.einsum('nc,nc->n', *[q, k]).unsqueeze(-1)
            l_neg = paddle.einsum('nc,ck->nk', *[q, self.queue.clone().detach()])
        else:
            l_pos = paddle.sum(paddle.multiply(q, k), axis=-1, keepdim=True)
            l_neg = paddle.matmul(q, self.queue.clone().detach())
        
        loss_generated = self.head(l_pos, l_neg, loss_dict=False)

        # for now we will not use queue mechanism
        # self._dequeue_and_enqueue((k_0 + k_1) / 2)

        return loss_generated, k
    

# ! for pinas
class ContrastiveHead(nn.Layer):
    '''Head for contrastive learning.
    '''

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg, loss_dict=False):
        '''
        Args:
            pos (Tensor): Nx1 positive similarity
            neg (Tensor): Nxk negative similarity
        '''
        logits = paddle.concat((pos, neg), axis=1)
        logits = logits / self.temperature
        # print('[DEBUG]logits', logits) # logits: [B, 1 + len_queue]
        labels = paddle.zeros((logits.shape[0], ), dtype=paddle.int64)
        loss = self.criterion(logits, labels)
        if loss_dict:
            losses = dict()
            losses['loss'] = loss
            return losses
        else:
            return loss
        



