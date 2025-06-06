from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import random


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
weak_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# 强增强
strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    RandAugment(n=2, m=10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img1 = weak_transform(img)
        img2 = strong_transform(img)
        return (img1, img2), label


def get_cifar10_dataloaders(labeled_indices, num_iters=20000, num_workers=2,
                            batch_size=64, test_batch_size=1024, mu=7):
    """
    获取FixMatch的CIFAR-10数据加载器
    mu: 无标签数据与有标签数据的比例
    """
    # 加载CIFAR-10数据集
    labeled_base_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=weak_transform
    )
    
    unlabeled_base_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    
    labeled_test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # 分割有标签和无标签数据
    all_indices = set(range(len(labeled_base_dataset)))
    unlabeled_indices = list(all_indices - set(labeled_indices))
    
    labeled_dataset = Subset(labeled_base_dataset, labeled_indices)
    unlabeled_dataset = Subset(unlabeled_base_dataset, unlabeled_indices)
    unlabeled_dataset = UnlabeledDataset(unlabeled_dataset)
    
    # 计算数据的批次大小
    labeled_batch_size = batch_size
    unlabeled_batch_size = batch_size * mu
    
    # 创建数据加载器
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_sampler=BatchSampler(
            RandomSampler(labeled_dataset, replacement=True, num_samples=labeled_batch_size*num_iters),
            labeled_batch_size,
            drop_last=True
        ),
        num_workers=num_workers,
        pin_memory=False
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_sampler=BatchSampler(
            RandomSampler(unlabeled_dataset, replacement=True, num_samples=unlabeled_batch_size*num_iters),
            unlabeled_batch_size,
            drop_last=True
        ),
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        labeled_test_dataset,
        batch_sampler=BatchSampler(
            RandomSampler(labeled_test_dataset, replacement=True, num_samples=len(labeled_test_dataset)),
            test_batch_size,
            drop_last=True
        ),
        num_workers=num_workers,
        pin_memory=False
    )
    
    return labeled_loader, unlabeled_loader, test_loader