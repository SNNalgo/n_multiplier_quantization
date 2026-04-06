import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(dataset="cifar10"):
    
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            # add some data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    if dataset == "imagenet":

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        trainset = torchvision.datasets.ImageFolder(root='/raid/ee-udayan/uganguly/data/ImageNet/train/', 
                        transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root='/raid/ee-udayan/uganguly/data/ImageNet/val/', 
                        transform=test_transform)
        
    return trainset, testset