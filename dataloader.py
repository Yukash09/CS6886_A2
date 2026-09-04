import torch 
import torch.nn as nn 
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader , random_split
from typing import Any

def get_test_data(batch_size: int = 128) :

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_data = datasets.CIFAR10(root="/kaggle/input/datasets/pankrzysiu/cifar10-python/" , train=False, download=False, transform=test_transform)
    print(f"Test Data len: {len(test_data)}")

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return test_loader 

def get_train_data(batch_size:int = 128) :

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data = datasets.CIFAR10(root="/kaggle/input/datasets/pankrzysiu/cifar10-python/", train=True, download=False, transform=train_transform)
    generator = torch.Generator().manual_seed(3)
    train_data , val_data = random_split(data, [int(0.85*len(data)) , len(data) - int(0.85*len(data))] , generator)

    print(f"Train and Val Data len: {len(train_data)} and {len(val_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data , batch_size=batch_size , shuffle=False)

    return train_loader , val_loader