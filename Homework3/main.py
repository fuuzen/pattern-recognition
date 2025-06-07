
import torch
import torchvision
import random
import argparse
import os
import numpy as np

from WideResNet import WideResNet
from get_cifar10_dataloaders import get_cifar10_dataloaders
import MixMatch
import FixMatch


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_labeled_subset(dataset, num_labeled_per_class):
    """创建平衡的有标签子集"""
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    labeled_indices = []
    
    for class_idx in range(10):  # CIFAR-10有10个类别
        class_indices = np.where(targets == class_idx)[0]
        selected_indices = np.random.choice(
            class_indices, num_labeled_per_class, replace=False
        )
        labeled_indices.extend(selected_indices)
    
    return labeled_indices


def draw(losses, accuracies, args):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt
    import os
    plt.figure(figsize=(10, 6))

    # 创建主轴用于绘制 loss
    ax1 = plt.gca()
    color = 'tab:red'
    ax1.set_xlabel(f'Iteration (×{args.eval_iter})')
    ax1.set_ylabel('Loss', color=color)
    line1 = ax1.plot(losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 创建次轴用于绘制 accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    line2 = ax2.plot(accuracies, color=color, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题
    plt.title('Training Loss and Test Accuracy')

    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    save_path = f'./images/{args.type}_{args.num_labels}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = WideResNet(depth=28, widen_factor=2, num_classes=10).to(device)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,}')

    # 使用算法
    if args.type == 'mixmatch':
        train = MixMatch.train
        mu = 1
        print('使用算法: mixmatch')
    elif args.type == 'fixmatch':
        train = FixMatch.train
        mu = 7
        print('使用算法: fixmatch')
    else:
        print('type has to be mixmatch or fixmatch !')
        exit(1)
    
    # 准备数据（FixMatch通常使用更少的标签数据）
    print('准备数据集...')
    temp_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    labeled_indices = create_balanced_labeled_subset(temp_dataset, num_labeled_per_class=args.num_labels)
    labeled_loader, unlabeled_loader, test_loader = get_cifar10_dataloaders(labeled_indices, args.num_iters, mu=mu)
    
    print(f'有标签样本数: {len(labeled_indices)}')
    print(f'无标签数据批次大小: {64 * mu}')
    
    # 训练模型
    print('开始训练...')
    train_losses, test_accuracies, optimizer = train(
        model,
        labeled_loader, unlabeled_loader, test_loader,
        num_iters=args.num_iters,
        device=device,
        eval_iter=args.eval_iter
    )

    model.save(f'./model/{args.type}_{args.num_labels}.pth', optimizer=optimizer)

    if args.draw:
        draw(train_losses, test_accuracies, args)

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)  # 40, 250, 4000
    parser.add_argument('-ni', '--num_iters', type=int, default=20000)  # 实验要求
    parser.add_argument('-ei', '--eval_iter', type=int, default=50)
    parser.add_argument('-t', '--type', type=str, default='mixmatch')
    parser.add_argument('-d', '--draw', type=bool, default=True)
    args = parser.parse_args()
    main(args)