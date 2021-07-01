# n_gpu=${1}
#
# spring.submit arun  -n$1 --job-name=multi-label --gpu \
#   "python train_bce.py \
#   -image_path /mnt/lustre/yankun/data/coco2014 \
#   -save_path /mnt/lustre/yankun/orderless-rnn-classification/save_path1"

n_gpu=${1}

job_name='train_bce'

GLOG_vmodule=MemcachedClient=-1 \
  srun --mpi=pmi2 -p vi_irdc --gres=gpu:${n_gpu} \
  --job-name=${job_name} \
  python -u train.py --logdir checkpoints/feature --batch_size 128 --top_rnns --lr 1e-4 --n_epochs 30
