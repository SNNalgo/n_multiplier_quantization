import ffcv
import torch
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
import matplotlib.pyplot as plt

from loader_ffcv import *

train_dataset = "/raid/ee-udayan/uganguly/data/ImageNet_ffcv/train_400_0.1_90.ffcv"
test_dataset = "/raid/ee-udayan/uganguly/data/ImageNet_ffcv/val_400_0.1_90.ffcv"

batch_size_train = 256
batch_size_test = 256
num_workers = 12
device = torch.device("cuda")
 
trainloader = create_train_loader_imagenet(train_dataset=train_dataset, num_workers=num_workers, batch_size=batch_size_train, device=device, distributed=False, in_memory=True)
testloader = create_val_loader_imagenet(val_dataset=test_dataset, num_workers=num_workers, device=device, batch_size=batch_size_test, distributed=False)

# iterate through trainloader once
for i, (images, labels) in enumerate(trainloader):
    if i == 0:
        # show image
        print(images[0].shape)
        plt.imsave("image.png", (images[10].permute(1, 2, 0)/255).cpu().numpy())
        break
