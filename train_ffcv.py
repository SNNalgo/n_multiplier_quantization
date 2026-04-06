import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from models import *
import wandb
from utils import *
from make_dataset import *
import resnet_act
import resnet_cifar_act
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import \
    RandomResizedCropRGBImageDecoder
from ffcv.transforms import *
from loader_ffcv import *
from torch.cuda.amp import autocast
import torchmetrics

wandb.init(project="ffcv train")

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--model', default='resnet18', type=str, help='model to be used for training the linear classifier')
parser.add_argument('--batch_size_train', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=90, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--num_workers', default=12, type=int, help='number of workers for loading the data')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--save_model', action='store_true', help='to save?')
parser.add_argument('--model_path', default='./saved_models/', help='path to save the model')
parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')

args = parser.parse_args()

wandb.config.update(args)

def main():

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.dataset == 'cifar10':
        num_classes = 10

        # train_dataset = "/raid/ee-udayan/uganguly/data/cifar_train.beton"
        # test_dataset = "/raid/ee-udayan/uganguly/data/cifar_test.beton"

        # trainloader = create_train_loader_cifar(train_dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size_train, device=args.device, distributed=False, in_memory=True)
        # testloader = create_test_loader_cifar(test_dataset=test_dataset, num_workers=args.num_workers, device=args.device, batch_size=args.batch_size_test, distributed=False)
        trainset, testset = get_dataset(args.dataset)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers, drop_last=False)

    if args.dataset == 'imagenet':
        num_classes = 1000
        train_dataset = "/raid/ee-udayan/uganguly/data/ImageNet_ffcv/train_500_0.50_90.ffcv"
        test_dataset = "/raid/ee-udayan/uganguly/data/ImageNet_ffcv/val_500_0.50_90.ffcv"

        # train_dataset = "/raid/ee-udayan/uganguly/data/ImageNet_ffcv/train_400_0.1_90.ffcv"
        # test_dataset = "/raid/ee-udayan/uganguly/data/ImageNet_ffcv/val_400_0.1_90.ffcv"

        trainloader = create_train_loader_imagenet(train_dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size_train, device=args.device, distributed=False, in_memory=True)
        testloader = create_val_loader_imagenet(val_dataset=test_dataset, num_workers=args.num_workers, device=args.device, batch_size=args.batch_size_test, distributed=False)
 
        # trainset, testset = get_dataset(args.dataset)

        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers, drop_last=False)

    print("length of trainloader: ", len(trainloader))
    print("length of testloader: ", len(testloader))

    if args.model == 'vgg13':
        model = VGG13_Act(affine=False, bias=True).to(args.device)

    if args.model == 'resnet18':
        #model = resnet_act.resnet18().to(args.device)
        model = torchvision.models.resnet18().to(args.device)

    if args.model == 'resnet20':
        model = resnet_cifar_act.resnet20().to(args.device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    loss_history = []
    top1_acc_history = []
    top5_acc_history = []
    acc_historty = []

    top1_acc_compute = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    top5_acc_compute = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)

    for epoch in range(args.num_epochs): 

        running_loss = 0.0
        model.train()

        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device) 

            optimizer.zero_grad()

            with autocast():
                output = model(inputs)
                loss = criterion(output, labels)

            loss.backward()
    
            optimizer.step()

            running_loss += loss.detach().cpu().item()

            wandb.log({"train_loss": loss.detach().cpu().item()})

        scheduler.step()

        #print epoch loss
        loss_history.append(running_loss / len(trainloader))
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            with autocast():

                for i, (inputs, labels) in enumerate(testloader, 0):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    output = model(inputs)

                    # output += model(torch.flip(inputs, dims=[3]))
                    # output /= 2
                                        # calculate accuracy
                    total += labels.size(0)
                    correct += (output.max(1)[1] == labels).sum().item()

                    top1_acc = top1_acc_compute.update(output, labels)
                    top5_acc = top5_acc_compute.update(output, labels)

                accuracy_test = 100 * correct / total

                top1_acc = top1_acc_compute.compute()*100
                top5_acc = top5_acc_compute.compute()*100

                top1_acc_history.append(top1_acc)
                top5_acc_history.append(top5_acc)

                max_top1_acc = max(top1_acc_history)
                max_top5_acc = max(top5_acc_history)

                wandb.log({"test_accuracy": accuracy_test})
                wandb.log({"top1 acc": top1_acc, "max top1 acc": max_top1_acc, "top5 acc": top5_acc, "max top5 acc": max_top5_acc})
                
                print("Epoch: ", epoch + 1)
                print('Top-1 accuracy: %.3f %%' % (top1_acc))
                print('Top-5 accuracy: %.3f %%' % (top5_acc))
                print('Max top-1 accuracy: %.3f %%' % (max_top1_acc))
                print('Max top-5 accuracy: %.3f %%' % (max_top5_acc))
            
    if args.save_model:
        torch.save(model.state_dict(), "./saved_models/" + str(args.dataset) + '_ffcv_' + str(args.model) + '_' + str(args.num_epochs) + '_' + 'non_quantized' + ".ckpt")

if __name__ == '__main__':
    main()