import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from models import *
import wandb
import torch.quantization as quantization


wandb.init(project="quant aware training")

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--batch_size_train', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading the data')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
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

    print(len(trainset))
    print(len(testset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers, drop_last=False)

    model = VGG13(affine=False, bias=True).to(args.device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)


    loss_history = []
    accuracy_test_history = []

    for epoch in range(args.num_epochs): 

        running_loss = 0.0
        model.train()

        for i, (inputs, labels) in enumerate(trainloader, 0):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

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

            for i, (inputs, labels) in enumerate(testloader, 0):

                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)

                # calculate accuracy
                total += labels.size(0)
                correct += (output.max(1)[1] == labels).sum().item()

            accuracy_test = 100 * correct / total

            accuracy_test_history.append(accuracy_test)
            max_accuracy = max(accuracy_test_history)


            wandb.log({"test_acc": accuracy_test, "max_test_acc": max_accuracy})

            print('Test accuracy: %.3f %%' % (accuracy_test))
            print('Max test accuracy: %.3f %%' % (max_accuracy))
         
    if args.save_model:
        torch.save(model.state_dict(), args.model_path + "vgg13_cifar10_normal_no_affine.ckpt")

if __name__ == '__main__':
    main()