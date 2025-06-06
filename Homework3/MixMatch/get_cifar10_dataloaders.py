from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


class UnlabeledDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return (img1, img2), label


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
    

def get_cifar10_dataloaders(labeled_indices, batch_size=64, num_workers=2):
    """获取CIFAR-10数据加载器"""
    # 加载CIFAR-10数据集
    labeled_base_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
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
    unlabeled_dataset = UnlabeledDataset(unlabeled_subset, train_transform)
    
    # 创建数据加载器
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    return labeled_loader, unlabeled_loader, test_loader
