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

DATASET_PATH="/datasets_local/ImageNet/train"
EXPERIMENT_PATH="/root/local_storage/swav/deepclusterv2_comp_learning_pretrain_128"
mkdir -p $EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=1 main_deepclusterv2.py \
--data_path $DATASET_PATH \
--nmb_crops 2 4 \
--size_crops 160 96 \
--min_scale_crops 0.08 0.05 \
--max_scale_crops 1. 0.14 \
--subset 260 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--feat_dim 128 \
--nmb_prototypes 1024 1024 1024 \
--compressive_clustering \
--epochs 50 \
--batch_size 32 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 300000 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet18 \
--dump_path $EXPERIMENT_PATH \
--workers 8 \
