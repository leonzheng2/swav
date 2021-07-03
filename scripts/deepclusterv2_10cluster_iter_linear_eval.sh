IMAGENET="/datasets_local/ImageNet"
EXPERIMENT_PATH="/root/local_storage/swav/deepclusterv2_10cluster_iter_linear_eval"
PRETRAINED="/root/local_storage/swav/deepclusterv2_10cluster_iter_pretrain/checkpoint.pth.tar"

mkdir -p $EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
--data_path $IMAGENET \
--pretrained $PRETRAINED \
--dump_path $EXPERIMENT_PATH \
--arch resnet18 \
--epochs 10
