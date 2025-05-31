#!/bin/bash

# Run DFME on CIFAR10 with different student models
cd DFME || exit

DATASET="cifar10"
DEVICE=0
GRAD_M=1
QUERY_BUDGET=20
LOG_DIR="log"
LR_G=1e-5
LOSS="l1"

for STUDENT_MODEL in resnet18_8x mobilenet_v2 densenet121; do
  echo "Running DFME on $DATASET with student model: $STUDENT_MODEL"
  python train_ensemble.py \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --grad_m "$GRAD_M" \
    --query_budget "$QUERY_BUDGET" \
    --log_dir "$LOG_DIR" \
    --lr_G "$LR_G" \
    --student_model "$STUDENT_MODEL" \
    --loss "$LOSS"
done
