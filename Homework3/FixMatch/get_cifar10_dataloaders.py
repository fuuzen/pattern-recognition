from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


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

def get_cifar10_dataloaders(labeled_indices, batch_size=64, mu=7, num_workers=2):
    """
    获取FixMatch的CIFAR-10数据加载器
    mu: 无标签数据与有标签数据的比例
    """
    
    # 标准增强（用于有标签数据和测试）
    standard_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 标准增强（用于测试）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 创建FixMatch数据集（弱增强）
    weak_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 创建FixMatch数据集（强增强）
    strong_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    # 加载CIFAR-10数据集
    labeled_base_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=standard_transform
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