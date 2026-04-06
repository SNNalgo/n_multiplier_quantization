import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import List
import numpy as np

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, Convert, Cutout, RandomTranslate
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.fields.decoders import SimpleRGBImageDecoder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])*255
IMAGENET_STD = np.array([0.229, 0.224, 0.225])*255
DEFAULT_CROP_RATIO = 224/256

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

def create_train_loader_imagenet(train_dataset, num_workers, batch_size, device, distributed, in_memory):
        
        #this_device = f'cuda:{self.gpu}'
        this_device = device
        train_path = Path(train_dataset)
        assert train_path.is_file()

        decoder = RandomResizedCropRGBImageDecoder((224, 224))

        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        #order = OrderOption.RANDOM

        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

def create_val_loader_imagenet(val_dataset, num_workers, device, batch_size, distributed):
        #this_device = f'cuda:{self.gpu}'
        this_device = device
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (256, 256)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

def create_train_loader_cifar(train_dataset, num_workers, batch_size, device, distributed, in_memory):
        
        this_device = device
        train_path = Path(train_dataset)
        assert train_path.is_file()

        image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(4, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            Squeeze()
        ]

        order = OrderOption.RANDOM

        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

def create_test_loader_cifar(test_dataset, num_workers, device, batch_size, distributed):
        
        this_device = device
        test_path = Path(test_dataset)
        assert test_path.is_file()

        image_pipeline: List[Operation] = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            Squeeze()
        ]

        order = OrderOption.RANDOM

        loader = Loader(test_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader
