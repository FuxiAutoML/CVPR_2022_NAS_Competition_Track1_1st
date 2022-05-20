cmd="python3 -u train_supernet_pinas_stage2_simple.py run
  --backbone resnet48
  --max_epoch 5
  --batch_size 256
  --lr 0.0005
  --warmup 5
  --dyna_batch_size 4
  --sample_method random
  --train_flag True
  --use_ckpt True
  --use_loss_expand False
  --bias_sample_flag True
  --stage1_sample_num 400
  --pretrained ILSVRC2012/ckpt/resnet48.pdparams
  --image_dir ILSVRC2012/
  --ckpt_dir ckpt/neo/s1/wholenet_May11_again.2/0.pdparams
  --save_dir ckpt/neo/s2/wholenet_May11_E0_simple_nofrozen.7
  --save_freq 1
  --log_freq 50
"
image="mirrors.xxx.com/autopooling_train/paddle-gpu2.2-cuda10.1-cudnn7:v1"
run_docker.sh "$image" "$cmd" > train_pinas_neo_s2_lab.log 2>&1
