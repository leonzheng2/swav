# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=deepclusterv2_400ep_pretrain
#SBATCH --time=25:00:00
#SBATCH --mem=450G

EXPERIMENT_PATH="/root/local_storage/swav/deepclusterv2_random_data_unbalanced_2"
IMAGENET="/datasets_local/ImageNet"
IMAGENET_TRAIN="${IMAGENET}/train"

PRETRAIN_EXPERIMENT_PATH="${EXPERIMENT_PATH}/pretrain"
PRETRAINED="${PRETRAIN_EXPERIMENT_PATH}/checkpoint.pth.tar"
LINEAR_EVAL_EXPERIMENT_PATH="${EXPERIMENT_PATH}/linear_eval"

mkdir -p $PRETRAIN_EXPERIMENT_PATH
mkdir -p $LINEAR_EVAL_EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=1 main_deepclusterv2.py \
--data_path $IMAGENET_TRAIN \
--nmb_crops 2 4 \
--size_crops 160 96 \
--min_scale_crops 0.08 0.05 \
--max_scale_crops 1. 0.14 \
--subset 720 \
--ratio_minority_class 0.8 \
--ratio_step_size 5 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--feat_dim 128 \
--nmb_prototypes 1024 1024 1024 \
--nmb_kmeans_iters 0 \
--epochs 50 \
--batch_size 32 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 300000 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet18 \
--dump_path $PRETRAIN_EXPERIMENT_PATH \
--workers 8 \

python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
--data_path $IMAGENET \
--pretrained $PRETRAINED \
--dump_path $LINEAR_EVAL_EXPERIMENT_PATH \
--arch resnet18 \
--workers 8 \
--epochs 10
