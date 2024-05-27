import logging
import os.path

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_metric_learning.samplers import MPerClassSampler

from root import ROOT_DIR
from augm.cutout import Cutout


def get_data_loaders(dataset, batch_size, num_workers=12, download=True, samples_per_class=10, base_dir=ROOT_DIR, per_sampler=True, use_augment=True):
    print("Loading toy dataset")
    logging.info(f"Loading toy dataset")
    trainset, testset = get_toy_datasets(dataset, download, base_dir=base_dir, use_augment=use_augment)
    if per_sampler:
        train_sampler = MPerClassSampler(trainset.targets, samples_per_class, None, len(trainset.targets))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_toy_datasets(dataset, download=True, base_dir=ROOT_DIR, use_augment=True):
    to_save_dir = os.path.join(base_dir, f"datasets/toy/{dataset}")
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            Cutout(0.5, 8),
            transforms.ToTensor()
        ])
    elif dataset == "cifar100":
        if use_augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                Cutout(0.5, 8),
                transforms.ToTensor()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=to_save_dir, train=True, download=download, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=to_save_dir, train=False, download=download, transform=test_transform)
    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=to_save_dir, train=True, download=download, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root=to_save_dir, train=False, download=download, transform=test_transform)
    else:
        raise ValueError(f"invalid dataset: {dataset}")
    return trainset, testset


def get_face_loaders(batch_size, base_dir=ROOT_DIR, num_workers=12, samples_per_class=5):
    print(f"Loading from {base_dir}")
    trainset, testset = get_face_datasets(base_dir)
    train_sampler = MPerClassSampler(trainset.targets, samples_per_class, batch_size, len(trainset.targets))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_face_datasets(base_dir=ROOT_DIR):
    train_folder = os.path.join(base_dir, "datasets/vggface2_test_al")
    test_folder = os.path.join(base_dir, "datasets/vggface2_test_al_test")
    print(f"Train folder: {train_folder}")
    print(f"Test folder: {test_folder}")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_folder, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_folder, transform=test_transform)
    logging.info(f"Loaded face dataset")
    return train_dataset, test_dataset