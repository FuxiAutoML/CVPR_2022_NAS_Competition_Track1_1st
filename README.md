# 代码使用方法
注意：需要准备一个具备paddle 2.2版本的docker镜像（paddle-gpu2.2-cuda10.1-cudnn7:v1）。若不使用docker环境，则应将下述.sh文件中的命令（cmd=...）在符合paddle 2.2的环境中运行。

## 训练：阶段一
sh local_train_pinas1_wholenet.sh

会保存多轮ckpt，此处建议使用首轮保存的ckpt，交给下一轮继承训练

py文件入口：train_supernet_pinas_stage1.py

## 训练：阶段二
sh local_train_pinas2_simple_wholenet_nof.sh

会保存多轮ckpt，此处建议使用首轮保存的ckpt，交给测试（eval）任务

py文件入口：train_supernet_pinas_stage2_simple.py

## 测试：阶段三
sh local_eval.sh

使用阶段二的ckpt（0.pdparams）文件，进行4.5万个子网结构acc预测

py文件入口：train_supernet.py

# 注意
代码中部分绝对路径需要修改为你运行时的具体文件路径，比如hapi_wrapper.py中的modelarch_file等。
