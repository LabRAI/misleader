<h1 align="center"> MISLEADER: Defending against Model Extraction with Ensembles of Distilled Models </h1>

# Introduction
This is the open-source code for our paper **MISLEADER: Defending against Model Extraction with Ensembles of Distilled Models**

## 📦 Package Requirements

To ensure compatibility, please use the following package versions:

| Package        | Version       | Notes                 |
|----------------|---------------|------------------------|
| Python         | 3.11          |                        |
| torch          | 2.2.1         | + CUDA 12.1            |
| torchvision    | 0.17.1        |                        |
| torchaudio     | 2.2.1         |                        |
| torchtext      | 0.17.1        |                        |
| torch-cuda     | 12.1          | For CUDA acceleration  |
| numpy          | 1.26.4        |                        |
| scikit-learn   | 1.1.2         |                        |
| scipy          | 1.15.2        |                        |
| matplotlib     | 3.8.4         |                        |
| pandas         | 2.2.2         |                        |
| tqdm           | 4.66.4        |                        |
| tensorboard    | 2.16.2        | For training visualization |
| protobuf       | 5.29.4        |                        |
<!-- | wandb          | 0.17.0        | Optional for logging   -->


## Train Teacher Model

Run the following command to train the teacher models

```
bash teacher.sh
```

Exemplary results:
```
Training teacher model on cifar10...
Files already downloaded and verified
Files already downloaded and verified
Train - Epoch 1, Batch: 1, Loss: 3.803519
Test Avg. Loss: 0.016970, Accuracy: 0.374500
Train - Epoch 2, Batch: 1, Loss: 1.691090
Test Avg. Loss: 0.014026, Accuracy: 0.481300
...
```

The teacher models will be saved in the "./Defense/cache/teacher/" directory

## Train Defense Ensemble Model

Run the following command to train the emsemble of defense models (for cifar10):

```
bash train.sh
```

Exemplary results:
```
Training defense model: resnet18_8x with λ=0.001, T=6, α=0.3
Files already downloaded and verified
Files already downloaded and verified
loading resnet18_8X
loading resnet18_8X
Total number of epochs: 261
Cost per iteration: 1536
Query Budget: 20000000
Defense Training Epoch 0: 100%|███████████████████████████████████████████| 196/196 [00:25<00:00,  7.64it/s, Total Loss=2.9920, KL Loss=2.9929, Atk Loss=0.8740]
...
Combining trained models using ensemble.py...
...
```
The defense models will be saved in the "./Defense/cache/defense/" directory, the emsemble of defense models will be saved in the "./Defense/ensemble/" directory.

## DFME Performance

Move the trained defense models and ensemble to "./DFME/ensemble/" directory.
Run the following command to train the emsemble of defense models

```
bash test.sh
```

Training log will be saved in the "./DFME/logs/" directory.
The real-time experimental results will be printed in the terminal like this:

```
Running DFME on cifar10 with student model: resnet18_8x
torch version 2.2.1
new cifar10 config
Namespace(batch_size=2048, query_budget=20000000, epoch_itrs=50, g_iter=1, d_iter=5, lr_S=0.1, lr_G=1e-05, nz=256, log_interval=10, loss='l1', scheduler='multistep', steps=[0.1, 0.3, 0.5], scale=0.3, dataset='cifar10', data_root='data', model='lenet5', weight_decay=0.0005, momentum=0.9, no_cuda=False, seed=53860, ckpt='checkpoint/teacher/teacher_cifar10.pt', student_load_path=None, model_id='debug', device=2, log_dir='log5', approx_grad=1, grad_m=1, grad_epsilon=0.001, forward_differences=1, no_logits=1, logit_correction='mean', rec_grad_norm=1, MAZE=0, store_checkpoints=1, student_model='resnet18_8x')
log5
Namespace(batch_size=2048, query_budget=20000000, epoch_itrs=50, g_iter=1, d_iter=5, lr_S=0.1, lr_G=1e-05, nz=256, log_interval=10, loss='l1', scheduler='multistep', steps=[0.1, 0.3, 0.5], scale=0.3, dataset='cifar10', data_root='data', model='lenet5', weight_decay=0.0005, momentum=0.9, no_cuda=False, seed=53860, ckpt='checkpoint/teacher/teacher_cifar10.pt', student_load_path=None, model_id='debug', device=2, log_dir='log5', approx_grad=1, grad_m=1, grad_epsilon=0.001, forward_differences=1, no_logits=1, logit_correction='mean', rec_grad_norm=1, MAZE=0, store_checkpoints=1, student_model='resnet18_8x', model_dir='checkpoint/student_debug')
Files already downloaded and verified
Files already downloaded and verified
loading resnet18_8X
using mobilenet_v2
Using densenet121 as the teacher network
Teacher restored from ./ensemble/cifar10.pth

Teacher - Test set: Accuracy: 9538/10000 (95.3800%)

loading resnet18_8X
channels 3
number of parameters student network 11173962

Total budget: 20000k
Cost per iterations:  14336
Total number of epochs:  28
Learning rate scheduling at steps:  [2, 8, 14]

Train Epoch: 1/28 [0/50 (0%)]   G_Loss: -0.257610 S_loss: 1.957680
Train Epoch: 1/28 [10/50 (20%)] G_Loss: -0.965819 S_loss: 0.111939
Train Epoch: 1/28 [20/50 (40%)] G_Loss: -0.074268 S_loss: 0.075729
Train Epoch: 1/28 [30/50 (60%)] G_Loss: -0.073817 S_loss: 0.075170
Train Epoch: 1/28 [40/50 (80%)] G_Loss: -0.076166 S_loss: 0.077855

 MAZE: 0, Test set cifar10: Average loss: 2.4544, Accuracy: 894/10000 (8.9400%), teacher resnet34_8x, student resnet18_8x
 ...
 ```

The final results are shown in the paper.


