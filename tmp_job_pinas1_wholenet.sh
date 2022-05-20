# ! main script
FLAGS_selected_gpus=0 python3 -u ttpathtt/ttjob_filett run \
  --backbone resnet48pinasv2 \
  --max_epoch 10 \
  --batch_size 256 \
  --lr 0.0008 \
  --warmup 1 \
  --dyna_batch_size 0 \
  --sample_method random \
  --train_flag True \
  --use_ckpt False \
  --use_tensor_dataset True \
  --pretrained ILSVRC2012/ckpt/resnet48.pdparams \
  --image_dir ILSVRC2012/ \
  --ckpt_dir ckpt/supernet_split/final.pdparams \
  --save_dir_new ckpt/neo/s1/wholenet_May11_again.2 \
  --largest_net_in_stage1 True \
  --num_workers 8 \
  --save_freq 1 \
  --log_freq 50

# ! pinas stage1用单卡才能保证loss正常
# 在此warmup实际无意义
