from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import random


class FixMatchDataset(Dataset):
    """FixMatch专用数据集，返回弱增强和强增强版本"""
    def __init__(self, dataset, weak_transform, strong_transform):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        weak_aug = self.weak_transform(img)
        strong_aug = self.strong_transform(img)
        return (weak_aug, strong_aug), label


ops = [
    lambda x: functional.rotate(x, random.uniform(-30, 30)),
    lambda x: functional.adjust_brightness(x, random.uniform(0.5, 1.5)),
    lambda x: functional.adjust_contrast(x, random.uniform(0.5, 1.5)),
    lambda x: functional.adjust_saturation(x, random.uniform(0.5, 1.5)),
]


class RandAugment:
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        
    def __call__(self, img):
        for _ in range(self.n):
            op = random.choice(ops)
            img = op(img)
        return img


# 标准增强（用于测试）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# 弱增强
weak_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# 强增强
strong_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    RandAugment(n=2, m=10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_cifar10_dataloaders(labeled_indices, batch_size=64, mu=7, num_workers=2):
    """
    获取FixMatch的CIFAR-10数据加载器
    mu: 无标签数据与有标签数据的比例
    """
    # 加载CIFAR-10数据集
    labeled_base_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=weak_aug
    )
    
    unlabeled_base_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # 分割有标签和无标签数据
    all_indices = set(range(len(labeled_base_dataset)))
    unlabeled_indices = list(all_indices - set(labeled_indices))
    
    labeled_dataset = Subset(labeled_base_dataset, labeled_indices)
    unlabeled_subset = Subset(unlabeled_base_dataset, unlabeled_indices)
    
    unlabeled_dataset = FixMatchDataset(unlabeled_subset, weak_aug, strong_aug)
    
    # 计算无标签数据的批次大小
    unlabeled_batch_size = batch_size * mu
    
    # 创建数据加载器
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    return labeled_loader, unlabeled_loader, test_loader