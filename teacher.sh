#!/bin/bash

cd teacher || exit

for dataset in cifar10 cifar100 MNIST; do
  echo "Training teacher model on $dataset..."
  python teacher.py --dataset "$dataset"
done
