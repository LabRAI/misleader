import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from my_utils import *
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SoftVotingEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        for model in self.models:
            model.eval()  # freeze all models

    def forward(self, x):
        with torch.no_grad():
            outputs = [F.softmax(model(x), dim=1) for model in self.models]
            return torch.stack(outputs).mean(dim=0)

parser = argparse.ArgumentParser(description='Train Defense d(x) with Inner Attacker')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10', 'cifar100'])    
args = parser.parse_args()

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.dataset == 'MNIST':
    model_names = ['lenet5', 'lenet5half', 'lenet5fifth']
else:
    model_names = ['resnet18_8x', 'mobilenet_v2', 'densenet121']

batch_size = args.batch_size
set_seed(42)

# Load models
def load_models(model_names, args):
    models = []
    for name in model_names:
        model = get_classifier(args, name, pretrained=False, num_classes=args.num_classes).to(device)
        ckpt_path = f'./checkpoint/defense/defense_{args.dataset}_{name}.pth'
        # model = get_classifier(args, 'lenet5', pretrained=False, num_classes=args.num_classes).to(device)
        # ckpt_path = f'./checkpoint/teacher/teacher_MNIST.pth'
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
    return models

models_loaded = load_models(model_names, args)

if args.dataset == 'MNIST':
    data_train = MNIST(args.data,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]), download=True)
    data_test = MNIST(args.data,
                    train=False,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]), download=True)

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, shuffle=False, num_workers=8)

elif args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data, transform=transform_train, download=True)
    data_test = CIFAR10(args.data, train=False, transform=transform_test, download=True)

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=100, shuffle=False, num_workers=0)

elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])

    data_train = CIFAR100(args.data, transform=transform_train, download=True)
    data_test = CIFAR100(args.data, train=False, transform=transform_test, download=True)

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=128, shuffle=False, num_workers=0)


# Hard Voting
def hard_voting(models, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = [model(images) for model in models]
            preds = torch.stack([out.argmax(dim=1) for out in outputs])
            majority_votes = torch.mode(preds, dim=0).values
            correct += (majority_votes == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Soft Voting
def soft_voting(models, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = [F.softmax(model(images), dim=1) for model in models]
            avg_output = torch.stack(outputs).mean(dim=0)
            preds = avg_output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

ensemble_model = SoftVotingEnsemble(models_loaded)
torch.save(ensemble_model.state_dict(), f'./ensemble/{args.dataset}.pth')


def evaluate_on_random_subset(models, dataset, portion=0.8, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), int(len(dataset) * portion), replace=False)
    subset_loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=128, shuffle=False)
    
    acc_hard = hard_voting(models, subset_loader)
    acc_soft = soft_voting(models, subset_loader)
    return acc_hard, acc_soft

num_runs = 5
hard_accuracies = []
soft_accuracies = []

for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}...")
    acc_hard, acc_soft = evaluate_on_random_subset(models_loaded, data_test, seed=run + 123)
    hard_accuracies.append(acc_hard)
    soft_accuracies.append(acc_soft)
    # print(f"  Hard Voting: {acc_hard * 100:.2f}%")
    print(f"  Soft Voting: {acc_soft * 100:.2f}%")

hard_mean = np.mean(hard_accuracies) * 100
hard_std = np.std(hard_accuracies) * 100
soft_mean = np.mean(soft_accuracies) * 100
soft_std = np.std(soft_accuracies) * 100

print("\n===== Ensemble Evaluation Summary (Random Subsets) =====")
# print(f"Hard Voting Accuracy: {hard_mean:.2f} ± {hard_std:.2f}%")
print(f"Soft Voting Accuracy: {soft_mean:.2f} ± {soft_std:.2f}%")
