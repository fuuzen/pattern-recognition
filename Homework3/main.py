
import torch
import torchvision
import random
import argparse
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


def draw(losses, accuracies, eval_iter, save_path='./images/latest.jpg'):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel(f'Iteration (×{eval_iter})')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Test Accuracy')
    plt.xlabel(f'Iteration (×{eval_iter})')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi 控制分辨率，bbox_inches 防止截断
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
    train_losses, test_accuracies = train(
        model,
        labeled_loader, unlabeled_loader, test_loader,
        num_iters=args.num_iters,
        device=device,
        eval_iter=args.eval_iter
    )
    
    return model, train_losses, test_accuracies

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)  # 40, 250, 4000
    parser.add_argument('-ni', '--num_iters', type=int, default=20000)  # 实验要求
    parser.add_argument('-ei', '--eval_iter', type=int, default=50)
    parser.add_argument('-t', '--type', type=str, default='mixmatch')
    parser.add_argument('-d', '--draw', type=bool, default=True)
    args = parser.parse_args()
    model, losses, accuracies = main(args)
    if args.draw:
        draw(losses, accuracies, args.eval_iter)