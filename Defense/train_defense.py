import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from my_utils import *
from lenet import LeNet5
from tqdm import tqdm
import argparse
from itertools import cycle
from torch.cuda.amp import autocast, GradScaler

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Knowledge Distillation loss with temperature and soft/hard mix.
    """
    alpha = params.alpha
    T = params.T
    kd_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1).clamp(min=1e-10)
    ) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1. - alpha)
    return kd_loss

def eval_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return 100. * correct / total

def main():
    # --------- Argument Parser --------- #
    parser = argparse.ArgumentParser(description='Train Defense d(x) with Inner Attacker')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10', 'cifar100'])
    parser.add_argument('--data', type=str, default='cache/data/')
    parser.add_argument('--output_dir', type=str, default='checkpoint/defense/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--teacher_model', type=str, default='lenet5')
    parser.add_argument('--defense_model', type=str, default='lenet5')
    parser.add_argument('--attacker_model', type=str, default='lenet5')
    parser.add_argument('--L', type=float, default=1.0, help='Lipschitz constant for the inner min loss term')
    parser.add_argument('--lr_defense', type=float, default=0.1, help='Learning rate for defense model')
    parser.add_argument('--query_budget', type=float, default=2, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--d_iter', type=int, default=1)   # defense updates per iteration
    parser.add_argument('--a_iter', type=int, default=5)   # attacker updates per defense update
    
    parser.add_argument('--lambda_', type=float, default=0.5, help='Weight for the inner min loss term')
    parser.add_argument('--T', type=float, default=2, help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight for the knowledge distillation loss')

    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    set_seed(42)
    # --------- Device Setup --------- #

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --------- Dataset --------- #
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

        defense = get_classifier(args, args.defense_model, pretrained=False, num_classes=10).to(device)
        optimizer_defense = optim.SGD(defense.parameters(), lr=args.lr_defense, momentum=0.9, weight_decay=5e-4)

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

        defense = get_classifier(args, args.defense_model, pretrained=False, num_classes=10).to(device)
        optimizer_defense = optim.SGD(defense.parameters(), lr=args.lr_defense, momentum=0.9, weight_decay=5e-4)

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

        defense = get_classifier(args, args.defense_model, pretrained=False, num_classes=100).to(device)
        optimizer_defense = optim.SGD(defense.parameters(), lr=args.lr_defense, momentum=0.9, weight_decay=5e-4)


    # Set number of classes based on dataset
    if args.dataset == 'MNIST':
        args.num_classes = 10
    elif args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise ValueError("Unsupported dataset")


    # --------- Augmented Loader for Attacker (simulate Pg) --------- #
    if args.dataset == 'cifar10':
        attacker_aug = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset == 'cifar100':
        attacker_aug = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])
    elif args.dataset == 'MNIST':
        attacker_aug = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    if args.dataset == 'cifar10':
        attacker_data_train = CIFAR10(args.data, transform=attacker_aug, download=False)
    elif args.dataset == 'cifar100':
        attacker_data_train = CIFAR100(args.data, transform=attacker_aug, download=False)
    elif args.dataset == 'MNIST':
        attacker_data_train = MNIST(args.data, transform=attacker_aug, download=False)

    attacker_data_loader = DataLoader(attacker_data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    attacker_data_iter = iter(attacker_data_loader)

    # --------- Load Teacher Model --------- #
    if args.dataset == 'MNIST':
        args.teacher_model = 'lenet5'
        teacher = LeNet5().to(device)
        teacher.load_state_dict(torch.load(f'./checkpoint/teacher/teacher_{args.dataset}.pth', map_location=device))
    elif args.dataset in  ['cifar10', 'cifar100']:
        args.teacher_model = 'resnet34_8x'
        teacher = get_classifier(args, args.teacher_model, pretrained=False, num_classes=args.num_classes).to(device)
        teacher.load_state_dict(torch.load(f'./checkpoint/teacher/teacher_{args.dataset}.pth', map_location=device))
    teacher.eval()
    # --------- Initialize Attacker Model --------- #
    attacker = get_classifier(args, args.attacker_model, pretrained=False, num_classes=args.num_classes).to(device)
    optimizer_attacker = optim.SGD(attacker.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # --------- Cost and Epochs --------- #
    args.query_budget *=  10**6
    args.query_budget = int(args.query_budget)
    args.cost_per_iteration = args.batch_size * (args.d_iter + args.a_iter)
    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"Total number of epochs: {number_epochs}")
    print(f"Cost per iteration: {args.cost_per_iteration}")
    print(f"Query Budget: {args.query_budget}")
    
    scheduler_defense = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_defense, T_max=number_epochs)
    scheduler_attacker = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_attacker, T_max=number_epochs)
    
    optimizer_defense.zero_grad()
    optimizer_defense.step()

    optimizer_attacker.zero_grad()
    optimizer_attacker.step()

    scheduler_defense.step()
    scheduler_attacker.step()

    best_defense_acc = 0.0
    patience = max(5, int(0.5 * number_epochs))
    patience_counter = 0

    # --------- Training Loop --------- #

    scaler = GradScaler()  # For AMP training
    attacker_loader_cycled = cycle(attacker_data_loader)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(int(number_epochs)):
        defense.train()
        attacker.train()

        pbar = tqdm(data_train_loader, desc=f"Defense Training Epoch {epoch}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            images_attacker, _ = next(attacker_loader_cycled)
            images_attacker = images_attacker.to(device, non_blocking=True)

            # ---- Step 1: Update attacker (inner min) ----
            optimizer_attacker.zero_grad()
            for _ in range(args.a_iter):
                with autocast():
                    attacker_logits = attacker(images_attacker).log_softmax(dim=1)
                    defense_logits_attacker = defense(images_attacker).softmax(dim=1).detach()

                    attacker_loss = nn.functional.kl_div(
                        attacker_logits, defense_logits_attacker, reduction='batchmean'
                    )

                scaler.scale(attacker_loss).backward()
            scaler.unscale_(optimizer_attacker)  
            torch.nn.utils.clip_grad_norm_(attacker.parameters(), max_norm=5.0)
            scaler.step(optimizer_attacker)   
            
            # with torch.no_grad():
            with torch.no_grad():
                teacher_logits = teacher(images)
                
            with autocast():
                defense_logits = defense(images)
                kl_teacher = loss_fn_kd(defense_logits, labels, teacher_logits, args)

                # KL(f_s || d) on attacker data
                f_s_log_probs = nn.functional.log_softmax(attacker(images_attacker).detach(), dim=1)
                d_probs_attacker = nn.functional.softmax(defense(images_attacker), dim=1).clamp(min=1e-7, max=1.0)

                kl_attacker = nn.functional.kl_div(
                    f_s_log_probs, d_probs_attacker, reduction='batchmean'
                )

                total_loss = kl_teacher - args.lambda_ * kl_attacker
                
            pbar.set_postfix({
                'Total Loss': f'{total_loss.item():.4f}', 
                'KL Loss': f'{kl_teacher.item():.4f}',
                # 'Inner Min': f'{inner_min_loss.item():.4f}', 
                'Atk Loss': f'{kl_attacker.item():.4f}'
            })

            optimizer_defense.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer_defense)
            torch.nn.utils.clip_grad_norm_(defense.parameters(), max_norm=5.0)
            scaler.step(optimizer_defense)   

            # --- Shared scaler update (after both .step) ---
            scaler.update()

            # --- LR schedulers (after optimizer.step) ---
        scheduler_defense.step()
        scheduler_attacker.step()
            
        defense_acc = eval_model(defense, data_test_loader, device)
        attacker_acc = eval_model(attacker, data_test_loader, device)
        # print(f"Defense Accuracy: {defense_acc:.2f}%")
        # print(f"Attacker Accuracy: {attacker_acc:.2f}%")

        # # Early stopping check
        if defense_acc > best_defense_acc:
            best_defense_acc = defense_acc
            patience_counter = 0
            torch.save(defense.state_dict(), os.path.join(args.output_dir, f'defense_{args.dataset}_{args.attacker_model}.pth'))
            torch.save(attacker.state_dict(), os.path.join(args.output_dir, f'attacker_{args.dataset}_{args.attacker_model}.pth'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Defense training finished!")

    # --------- Final Evaluation --------- #
    def evaluate(model, loader, name="Model"):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        print(f"{name} Accuracy: {acc:.2f}%")
        return acc

    print("\n----- Final Evaluation -----")
    evaluate(teacher, data_test_loader, name="Teacher Model")
    evaluate(defense, data_test_loader, name="Defense Model")
    evaluate(attacker, data_test_loader, name="Final Inner Attacker")
    
if __name__ == '__main__':
    main()