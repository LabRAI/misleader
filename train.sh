#!/bin/bash

# Script to train ensemble defense models for CIFAR10

cd Defense || exit

# Common parameters
DATASET="cifar10"
QUERY_BUDGET=20
BATCH_SIZE=256
TEACHER_MODEL="resnet34_8x"

# Define experiments as: "lambda T alpha defense_model attacker_model"
experiments=(
  "0.001 6 0.3 resnet18_8x resnet18_8x"
  "0.05  4 0.9 mobilenet_v2 mobilenet_v2"
  "0.01  2 0.3 densenet121 densenet121"
)

for exp in "${experiments[@]}"; do
  read -r LAMBDA T ALPHA DEFENSE_MODEL ATTACKER_MODEL <<< "$exp"
  echo "Training defense model: $DEFENSE_MODEL with λ=$LAMBDA, T=$T, α=$ALPHA"

  python train_defense.py \
    --dataset "$DATASET" \
    --query_budget "$QUERY_BUDGET" \
    --batch_size "$BATCH_SIZE" \
    --lambda_ "$LAMBDA" \
    --T "$T" \
    --alpha "$ALPHA" \
    --teacher_model "$TEACHER_MODEL" \
    --defense_model "$DEFENSE_MODEL" \
    --attacker_model "$ATTACKER_MODEL"
done

# Run ensemble step after all trainings
echo "Combining trained models using ensemble.py..."
python ensemble.py
