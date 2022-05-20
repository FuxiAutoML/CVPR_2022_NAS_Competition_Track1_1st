python3 -um paddle.distributed.launch --gpu 0 cvpr_baseline/CVPR_2022_Track1_demo/pretrain.py run \
  --arch 1322221222220000122200000024540000000000005525000000 \
  --backbone resnet48 \
  --max_epoch 3 \
  --batch_size 1024 \
  --lr 0.001 \
  --warmup 2 \
  --dyna_batch_size 2 \
  --save_dir cvpr_baseline/CVPR_2022_Track1_demo/ckpt/ \
  --log_freq 10 \
