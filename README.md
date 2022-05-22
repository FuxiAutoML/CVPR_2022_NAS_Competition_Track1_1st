# CVPR2022 NAS竞赛Track 1第1名方案
B榜排名：第一

## 方案说明
本项目使用自监督+蒸馏学习为核心思路，将超网(supernet)分为两极端训练。其中，阶段一继承官方的resnet pretrain参数，参考对比学习(Contrastive Learning)的思路，进行自监督训练，目的在于使得resnet backbone网络学习到更好的特征表示能力；阶段二中，student网络将继承阶段一训练得到的网络参数，teacher仍然使用官方的pretrain参数，student网络将在每个batch内进行子网结构有偏采样，从而达到One-Shot Learning的目的。

### 核心概念
- 自监督对比学习：通过实验，我们发现官方pretrain参数对图像特征的表达能力有限，因此，参考[MOCO](https://arxiv.org/abs/1911.05722)等工作，我们首先将网络训练一个阶段。此阶段中，网络将通过MLP层把resnet卷积得到的feature map转换为feature embedding。此阶段的训练，将使得网络尽可能地拉进同一张图片得到的feature embedding，并尽可能拉远不同图片得到的feature embedding. 为了避免过拟合，我们对同一张图片做了两类不同的随机变换，从而提高此阶段训练的泛化能力。经过此阶段训练，网络相对于原始的pretrain参数，将获得更好的特征表示能力。
- 有偏采样：在第二阶段，我们继承上述自监督对比学习的checkpoint，并对supernet的subnet结构进行采样。在One-For-All思路下，较大的subnet必然有部分参数与较小的subnet重合。因此，如果使用普通的随机采样 （即每个batch中训练的子网结构随机选取），必然会导致小网络所包含的参数，训练更为频繁，而大网络所包含的参数，训练比较稀疏。为此，我们设计了有偏采样，将此阶段的训练偏向于选取较大的网络结构，从而避免小网络的参数被大网络影响。

### 重要文件说明：
- train_supernet_pinas_stage1.py: 阶段一训练入口
- train_supernet_pinas_stage2_simple.py: 阶段二训练入口
- train_supernet.py: 阶段三（即测试阶段）训练入口
- hnas/models/pinasv2.py: 阶段一对比学习使用的网络结构
- hnas/utils/hapi_wrapper_pinas.py: 阶段一对比学习使用的paddle hapi
- hnas/models/resnet.py: 阶段二、三使用的网络结构，与比赛官方resnet相同
- hnas/utils/SampleMethod.py: 阶段二子网有偏采样相关逻辑
- hnas/utils/hapi_wrapper.py： 阶段二、三使用的paddle hapi，其中也包含了阶段二中子网有偏采样的逻辑
### 超参配置
对实验效果影响较大的超参主要包括dyna_batch_size、learning_rate, batch_size, max_epoch等。最佳超参配置请参考以下文件中的内容：
- local_train_pinas1_wholenet.sh
- local_train_pinas2_simple_wholenet_nof.sh

## 代码使用方法
注意：需要准备一个具备paddle 2.2版本的docker镜像（paddle-gpu2.2-cuda10.1-cudnn7:v1）。若不使用docker环境，则应将下述.sh文件中的命令（cmd=...）在符合paddle 2.2的环境中运行。

### 训练：阶段一
sh local_train_pinas1_wholenet.sh

会保存多轮ckpt，此处建议使用首轮保存的ckpt，交给下一轮继承训练

py文件入口：train_supernet_pinas_stage1.py

### 训练：阶段二
sh local_train_pinas2_simple_wholenet_nof.sh

会保存多轮ckpt，此处建议使用首轮保存的ckpt，交给测试（eval）任务

py文件入口：train_supernet_pinas_stage2_simple.py

### 测试：阶段三
sh local_eval.sh

使用阶段二的ckpt（0.pdparams）文件，进行4.5万个子网结构acc预测

py文件入口：train_supernet.py

### 注意
代码中部分绝对路径需要修改为你运行时的具体文件路径，比如hapi_wrapper.py中的modelarch_file等。另外，.sh脚本中的文件路径配置，也需要更改为本地可用的路径，比如pretrained, image_dir等。

## checkpoint文件
在checkpoint_files目录下保存有多个ckpt，解压后使用(可通过local_eval脚本配置ckpt_dir变量来加载)。