import paddle
import paddle.nn as nn
from pprint import pprint
# from mmcv.cnn import kaiming_init, normal_init

debug_print = False

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
        labels = paddle.zeros((logits.shape[0], ), dtype=paddle.int64)
        loss = self.criterion(logits, labels)
        if loss_dict:
            losses = dict()
            losses['loss'] = loss
            return losses
        else:
            return loss
            
# class ClsHead(nn.Layer): # same as HeadBlock
#     """Simplest classifier head, with only one fc layer.
#     """
#     def __init__(self, inplanes, num_classes=1000):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
#         self.fc = nn.Linear(inplanes, num_classes)
    
#     def forward(self, x):
#         x = self.fc(self.avgpool(x).flatten(1))
#         return x

    # def init_weights(self, init_linear='normal', std=0.01, bias=0.):
    #     assert init_linear in ['normal', 'kaiming'], \
    #         "Undefined init_linear: {}".format(init_linear)
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             if init_linear == 'normal':
    #                 normal_init(m, std=std, bias=bias)
    #             else:
    #                 kaiming_init(m, mode='fan_in', nonlinearity='relu')
    #         elif isinstance(m,
    #                         (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
    #             if m.weight is not None:
    #                 nn.init.constant_(m.weight, 1)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    #     assert isinstance(x, (tuple, list)) and len(x) == 1
    #     x = x[0]
    #     if self.with_avg_pool:
    #         assert x.dim() == 4, \
    #             "Tensor must has 4 dims, got: {}".format(x.dim())
    #         x = self.avg_pool(x)
    #     x = x.reshape(x.shape[0], -1)
    #     cls_score = self.fc_cls(x)
    #     return [cls_score]

    # def loss(self, cls_score, labels):
    #     losses = dict()
    #     assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
    #     losses['loss'] = self.criterion(cls_score[0], labels)
    #     # losses['acc'] = accuracy(cls_score[0], labels)
    #     return losses


# def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
#     assert init_linear in ['normal', 'kaiming'], \
#         "Undefined init_linear: {}".format(init_linear)
#     for m in module.modules():
#         if isinstance(m, nn.Linear):
#             if init_linear == 'normal':
#                 normal_init(m, std=std, bias=bias)
#             else:
#                 kaiming_init(m, mode='fan_in', nonlinearity='relu')
#         elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
#                             nn.GroupNorm, nn.SyncBatchNorm)):
#             if m.weight is not None:
#                 nn.init.constant_(m.weight, 1)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

class NonLinearNeckV1(nn.Layer):
    '''The non-linear neck in MoCO v2: fc-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool

        weight_attr1 = paddle.framework.ParamAttr(initializer=nn.initializer.KaimingNormal())
        bias_attr1 = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0))
        weight_attr2 = paddle.framework.ParamAttr(initializer=nn.initializer.KaimingNormal())
        bias_attr2 = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0))

        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels, weight_attr=weight_attr1, bias_attr=bias_attr1), 
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels, weight_attr=weight_attr2, bias_attr=bias_attr2), 
        )

    # def init_weights(self, init_linear='normal'):
    #     _init_weights(self, init_linear)

    def forward(self, x):
        # assert len(x) == 1
        # x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x_reshaped = paddle.reshape(x, [x.shape[0], -1])
        out = self.mlp(x_reshaped)
        return out

from .builder import backbone

USE_PRELU = False

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


class StemBlock(nn.Layer):
    def __init__(self, inplanes=3, outplanes=64, kernel_size=7, stride=2, padding=3, bias_attr=False):
        super().__init__()
        self.conv = nn.Conv2D(inplanes, outplanes, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(outplanes)
        self.relu = nn.PReLU() if USE_PRELU else nn.ReLU() # parameter name is with variable definition name
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class HeadBlock(nn.Layer):
    def __init__(self, inplanes, num_classes=1000):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(inplanes, num_classes)
    
    def forward(self, x):
        x = self.fc(self.avgpool(x).flatten(1))
        return x
    

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        self.conv1 = nn.Conv2D(inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.PReLU() if USE_PRELU else nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.stride = stride

        if stride != 1 or inplanes != planes * BasicBlock.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2D(inplanes, planes * BasicBlock.expansion, 1, stride=stride, bias_attr=False),
                norm_layer(planes * BasicBlock.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        # in_shape = x.shape[1]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # mi_shape = out.shape[1]
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def reorder_weights(self):
        conv2_weights = self.conv2.parameters()[0]
        conv1_weights = self.conv1.parameters()[0]

        importance = paddle.sum(paddle.abs(conv2_weights), axis=[0, 2, 3])
        sorted_idx = paddle.argsort(importance, axis=0, descending=True)
        reorder_conv2_weights = paddle.index_select(conv2_weights, sorted_idx, axis=1)
        conv2_weights.set_value(reorder_conv2_weights)
        adjust_bn_according_to_idx(self.bn1, sorted_idx)
        conv1_weights.set_value(paddle.index_select(conv1_weights, sorted_idx, axis=0))

class BottleneckBlock(nn.Layer):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(width, width, 3, padding=dilation, stride=stride, groups=groups, dilation=dilation,
                               bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.PReLU() if USE_PRELU else nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def reorder_weights(self):
        conv3_weights = self.conv3.parameters()[0]
        conv2_weights = self.conv2.parameters()[0]
        conv1_weights = self.conv1.parameters()[0]

        importance = paddle.sum(paddle.abs(conv3_weights), axis=[0, 2, 3])
        sorted_idx = paddle.argsort(importance, axis=0, descending=True)
        reorder_conv3_weights = paddle.index_select(conv3_weights, sorted_idx, axis=1)
        conv3_weights.set_value(reorder_conv3_weights)
        adjust_bn_according_to_idx(self.bn2, sorted_idx)
        conv2_weights.set_value(paddle.index_select(conv2_weights, sorted_idx, axis=0))

        importance = paddle.sum(paddle.abs(conv2_weights), axis=[0, 2, 3])
        sorted_idx = paddle.argsort(importance, axis=0, descending=True)
        reorder_conv2_weights = paddle.index_select(conv2_weights, sorted_idx, axis=1)
        conv2_weights.set_value(reorder_conv2_weights)
        adjust_bn_according_to_idx(self.bn1, sorted_idx)
        conv1_weights.set_value(paddle.index_select(conv1_weights, sorted_idx, axis=0))


class ResNet(nn.Layer):
    """ResNet"""
    
    def __init__(self, block, layers=None, base_channels=None, num_classes=1000, pinas_stage=0):
        super(ResNet, self).__init__()
        
        self.num_classes = num_classes
        self._norm_layer = nn.BatchNorm2D

        stride_list = [1, 2, 2, 2]

        inplanes = base_channels[0]
        self.blocks = nn.LayerList([StemBlock(3, inplanes, 7, 2, 3, False)])
        
        for d, c, s in zip(layers, base_channels, stride_list):
            for _ in range(d):
                self.blocks.append(
                    block(inplanes, c, s, self._norm_layer)
                )
                s = 1
                if inplanes != c * block.expansion:
                    inplanes = c * block.expansion
                
        self.blocks.append(HeadBlock(base_channels[-1] * block.expansion, num_classes=1000))
        # self.original_head = HeadBlock(base_channels[-1] * block.expansion, num_classes=1000)
        self.pinas_stage = pinas_stage
        self.neck = NonLinearNeckV1(
                in_channels=base_channels[-1],
                hid_channels=base_channels[-1],
                out_channels=128,
                with_avg_pool=True
            ) # the so-called "MLP"                
        self.act_depth_list = layers

    def reorder(self):
        for b in self.blocks[1:-1]:
            b.reorder_weights()

    def forward(self, x):
        x = self.blocks[0](x) # stem conv
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.act_depth_list[stage_id]
            active_idx = block_idx[:depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)
        x = self.blocks[-1](x)
        return x
    
    def forward_backbone(self, x):
        x = self.blocks[0](x) # stem conv
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.act_depth_list[stage_id] # ! support for variable depth here
            active_idx = block_idx[:depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)
        return x
    
    def forward_neck(self, x):
        # print('[DEBUG]forward_neck called')
        x = self.neck(x)
        return x
    
    def forward_cls_head(self, x):
        x = self.blocks[-1](x)
        return x
    

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks[1:-1], 1):
            if isinstance(block.downsample, nn.Sequential) and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list


@backbone.register_module() # ! the api called by train_supernet
def resnet48pinasv2(pretrained=False, reorder=False, pinas_stage=0, **kwargs):
    """resnet"""
    net = ResNet(BasicBlock, layers=[5, 5, 8, 5], base_channels=[64, 128, 256, 512], num_classes=1000, pinas_stage=pinas_stage)
    if pretrained:
        net.set_state_dict(paddle.load(pretrained))
        if reorder:
            net.reorder()
    return net