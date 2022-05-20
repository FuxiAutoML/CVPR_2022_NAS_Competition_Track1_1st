path=$(pwd)
file=$1
run_docker.sh mirrors.xxx.com/autopooling_train/paddle2.3-cuda10.2-cudnn7:v1 "sh ${path}/${file}"
