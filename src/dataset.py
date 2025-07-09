from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import NamedTuple

class CIFAR10Reconstruction(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, _ = self.cifar10[idx]  # discard the label
        return image, image  # Image-Image reconstruction
    

class DatasetSplit(NamedTuple):
    train: Dataset
    val: Dataset

def create_datasets(data_path, feature_transform =  transforms.ToTensor(), target_transform= transforms.ToTensor()):
    train_data = CIFAR10Reconstruction(root=data_path, train=True, transform=feature_transform)
    val_data = CIFAR10Reconstruction(root=data_path, train=False, transform=target_transform)
    return DatasetSplit(train=train_data, val=val_data)

