import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from snn_models import *
from torch import autocast
import wandb
from make_dataset_snn import *
import torch.quantization as quantization
from snntorch import functional as SF
import warnings
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")

wandb.init(project="qat snn")

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--batch_size_train', default=128, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=128, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading the data')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for training the linear classifier')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--label_percent', default=100, type=float, help='percentage of labeled data to be used for training the linear classifier')
parser.add_argument('--dataset', default='dvsgestures', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--num_time_steps', default=10, type=int, help='number of time steps for the SNN') # if using fixed number of time steps
parser.add_argument('--use_time_window', action='store_true', help='use time window for the SNN')
parser.add_argument('--time_window', default=10, type=int, help='time window for the SNN')

parser.add_argument('--resize_dim', default=128, type=int, help='size of the input image')
parser.add_argument('--beta', default=0.5, type=float, help='beta for the loss function')
parser.add_argument('--save_model', action='store_true', help='save the model')
parser.add_argument('--model_path', default='/raid/ee-udayan/uganguly/raghav/quant_aware_training/saved_models_snn', type=str, help='path to save the model')   
parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')


args = parser.parse_args()

wandb.config.update(args)

def main():

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    trainloader, testloader = make_dataset_snn(dataset=args.dataset, use_time_window=args.use_time_window, time_window=args.time_window, num_time_steps=args.num_time_steps, label_percent=args.label_percent, batch_size_train=args.batch_size_train, 
                                               batch_size_test=args.batch_size_test, num_workers=args.num_workers, resize_dim=args.resize_dim)

    if args.dataset == 'dvsgestures':
        num_classes = 11
    elif args.dataset == 'ncars':
        num_classes = 2
    elif args.dataset == 'cifar10dvs':
        num_classes = 10
    elif args.dataset == 'ncaltech101':
        num_classes = 101

    model = SNN_VGG(beta=args.beta, num_classes=num_classes).to(args.device)

    criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = GradScaler()

    accuracy_test_history = []
    accuracy_train_history = []

    for epoch in range(args.num_epochs): 

        model.train()

        for i, (inputs, labels) in enumerate(trainloader, 0):
            
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(inputs)
                loss = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            wandb.log({"train_loss": loss.detach().cpu().item()})

        scheduler.step()

        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():

            # for i, (inputs, labels) in enumerate(trainloader, 0):

            #     labels = labels.type(torch.LongTensor)
            #     inputs = inputs.to(device)
            #     labels = labels.to(device)

            #     inputs = inputs.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

            #     outputs = model(inputs)

            #     # calculate number of correct predictions
            #     _, idx = outputs.sum(dim=0).max(1)
            #     correct += idx.eq(labels).sum().item()

            #     total += labels.size(0)

            # accuracy_train = 100. * correct / total
            # accuracy_train_history.append(accuracy_train)
            # max_accuracy = max(accuracy_train_history)

            # wandb.log({"train_acc": accuracy_train, "max_train_acc": max_accuracy})

            # print('Train accuracy: %.3f %%' % (accuracy_train))
            # print('Max train accuracy: %.3f %%' % (max_accuracy))


            for i, (inputs, labels) in enumerate(testloader, 0):

                labels = labels.type(torch.LongTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = inputs.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

                outputs = model(inputs)

                # calculate number of correct predictions
                _, idx = outputs.sum(dim=0).max(1)
                correct += idx.eq(labels).sum().item()

                total += labels.size(0)

            accuracy_test = 100. * correct / total
            accuracy_test_history.append(accuracy_test)
            max_accuracy = max(accuracy_test_history)

            wandb.log({"test_acc": accuracy_test, "max_test_acc": max_accuracy})

            print('Epoch: %d' % (epoch + 1))
            print('Test accuracy: %.3f %%' % (accuracy_test))
            print('Max test accuracy: %.3f %%' % (max_accuracy))
         
    if args.save_model:
        torch.save(model.state_dict(), args.model_path + "_vgg_" + args.dataset + "_snn.pt")

if __name__ == '__main__':
    main()