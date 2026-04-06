import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from models import *
import resnet_act
import resnet_cifar_act
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from loader_ffcv import *
import torchmetrics

wandb.init(project="quant aware training weights and activations")

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--batch_size_train', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading the data')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--epoch_log_test', default=1, type=int, help='number of epochs after which to log test accuracy')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--model', default='resnet20', type=str, help='model to be used for training the linear classifier')
parser.add_argument('--seed', default=1, type=int, help='seed for random number generator')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--lr_levels_w', default=1e-4, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--lr_levels_a', default=1e-4, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--reg_lambda_w', default=1e-2, type=float, help='lambda for quantization')
parser.add_argument('--reg_lambda_a', default=1e-2, type=float, help='lambda for quantization')

parser.add_argument('--num_bits', default=4, type=int, help='number of bits for quantization')
parser.add_argument('--w_limit', default=0.1, type=float, help='weight limit for quantization')
parser.add_argument('--a_limit', default=5, type=float, help='activation limit for quantization')
parser.add_argument('--sig_factor', default=1, type=float, help='sigma factor for quantization')

parser.add_argument('--use_pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--fixed_levels', action='store_true', help='use fixed quantization levels')
parser.add_argument('--learnable_levels', action='store_true', help='use learnable quantization levels')
parser.add_argument('--lambda_scheduler', action='store_true', help='use lambda scheduler')
parser.add_argument('--clamp_activations', action='store_true', help='clamp activations')
parser.add_argument('--quantization_aware', action='store_true', help='quantization aware training')    
parser.add_argument('--save_model', action='store_true', help='save model')

args = parser.parse_args()

wandb.config.update(args)

def calc_loss(p, l, sigma):
    return 1 - 1*torch.exp(-(p - l)**2/(sigma**2))

def get_closest_loss(p, levels, sigma):
    # takes in p which is a 3/4d tensor and levels which is a 1d tensor
    # returns loss and number of parameters
    for level in levels:
        if level == levels[0]:
            loss = calc_loss(p, level, sigma)
        else:
            loss = torch.min(loss, calc_loss(p, level, sigma))
    
    return torch.sum(loss), p.numel()

def get_closest_loss_variable_sigma(p, levels, sigma):
    # takes in p which is a 3/4d tensor and levels which is a 1d tensor
    # returns loss and number of parameters
    quant_step_array = torch.zeros(len(levels)-1).to(p.device)

    for i in range(len(levels)-1):
        quant_step_array[i] = levels[i+1] - levels[i]

    for level in levels:
        if level == levels[0]:
            loss = calc_loss(p, level, sigma)
        else:
            loss = torch.min(loss, calc_loss(p, level, sigma))
    
    return torch.sum(loss), p.numel()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.dataset == 'cifar10':
        num_classes = 10
        train_dataset = "/raid/ee-udayan/uganguly/data/cifar_train.beton"
        test_dataset = "/raid/ee-udayan/uganguly/data/cifar_test.beton"

        trainloader = create_train_loader_cifar(train_dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size_train, device=args.device, distributed=False, in_memory=True)
        testloader = create_test_loader_cifar(test_dataset=test_dataset, num_workers=args.num_workers, device=args.device, batch_size=args.batch_size_test, distributed=False)

        # train_transform = transforms.Compose([
        #     # add some data augmentation
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, 4),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        
        # test_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    if args.dataset == 'imagenet':
        num_classes = 1000
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
        
        

    print(len(trainloader))
    print(len(testloader))

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers, drop_last=False)


    if args.model == 'vgg13':
        model = VGG13_Act(affine=False, bias=True).to(args.device)
        model_copy = VGG13_Act(affine=False, bias=True).to(args.device)

        if args.use_pretrained:
            model.load_state_dict(torch.load("./saved_models/cifar10_vgg13_non_quantized.ckpt"))

    
    if args.model == 'resnet18':
        model = resnet_act.resnet18().to(args.device)
        model_copy = resnet_act.resnet18().to(args.device)


    if args.model == 'resnet20':

        model = resnet_cifar_act.resnet20().to(args.device)
        model_copy = resnet_cifar_act.resnet20().to(args.device)

        if args.use_pretrained:
            model.load_state_dict(torch.load("./saved_models/cifar10_resnet20_non_quantized.ckpt"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # defining weight parameters

    max_val_w = 2**args.num_bits - 1
    zero_pt_w = 2**(args.num_bits-1)-0.5
    quant_step_w = np.float32((2*args.w_limit)/max_val_w)
    reg_sig_w = args.sig_factor*quant_step_w

    max_val_a = 2**args.num_bits - 1
    quant_step_a = np.float32((args.a_limit)/max_val_a)
    reg_sig_a = args.sig_factor*quant_step_a

    reg_lambda_w = torch.tensor(args.reg_lambda_w).to(device)
    reg_lambda_a = torch.tensor(args.reg_lambda_a).to(device)

    if args.fixed_levels:

        allowed_weight = (np.arange(0, 2**args.num_bits) - zero_pt_w)*quant_step_w
        allowed_weight = torch.tensor(allowed_weight, dtype=torch.float32).to(device)

        allowed_act = (np.arange(0, 2**args.num_bits))*quant_step_a
        allowed_act = torch.tensor(allowed_act, dtype=torch.float32).to(device)

        print("allowed activations:", allowed_act)
        print("allowed weights:", allowed_weight)

    if args.learnable_levels:

        allowed_weight = (np.arange(0, 2**args.num_bits) - zero_pt_w)*quant_step_w
        allowed_weight = nn.Parameter(torch.tensor(allowed_weight, dtype=torch.float32).to(device))

        allowed_act = (np.arange(0, 2**args.num_bits))*quant_step_a
        allowed_act = nn.Parameter(torch.tensor(allowed_act, dtype=torch.float32).to(device))

        optimizer_levels_w = optim.SGD([allowed_weight], lr=args.lr_levels_w, momentum=args.momentum, weight_decay=0)
        scheduler_levels_w = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_levels_w, T_max=args.num_epochs)

        optimizer_levels_a = optim.SGD([allowed_act], lr=args.lr_levels_a, momentum=args.momentum, weight_decay=0)
        scheduler_levels_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_levels_a, T_max=args.num_epochs)

        print(allowed_weight)
        print(allowed_act)

    accuracy_test_history = []
    top_1_accuracy_history = [] 
    top1_acc_compute = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)


    for epoch in range(args.num_epochs): 

        running_loss = 0.0
        model.train()

        # make reg_sig gradually smaller, go from quant_step to 0.5*quant_step
        # reg_sig = quant_step - (quant_step - 0.5*quant_step) * (epoch / args.num_epochs)
        
        if args.lambda_scheduler:
            # go from lambda to 10*lambda 
            reg_lambda_w = args.reg_lambda_w - (args.reg_lambda_w - 10*args.reg_lambda_w) * (epoch / args.num_epochs)
            reg_lambda_a = args.reg_lambda_a - (args.reg_lambda_a - 10*args.reg_lambda_a) * (epoch / args.num_epochs)

        for i, (inputs, labels) in enumerate(trainloader, 0):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if args.learnable_levels:
                optimizer_levels_w.zero_grad()
                optimizer_levels_a.zero_grad()

            if args.clamp_activations:
                output, activations = model.forward_clamp(inputs, allowed_act)
            elif args.fixed_levels or args.learnable_levels:
                output, activations = model.forward_act(inputs)
            else:
                output = model(inputs)

            ce_loss = criterion(output, labels)

            if args.fixed_levels:
                
                reg_loss_w = torch.tensor(0).to(device)
                reg_loss_a = torch.tensor(0).to(device)

                for i, (names, params) in enumerate(model.named_parameters()):
                    if params.requires_grad and 'bn' not in names:
                        reg_loss_current, num_params_current = get_closest_loss(params, allowed_weight, reg_sig_w)
                        reg_loss_w += reg_loss_current/num_params_current
                
                for activation in activations:
                    reg_loss_current, num_params_current = get_closest_loss(activation, allowed_act, reg_sig_a)
                    reg_loss_a += reg_loss_current/num_params_current

            if args.learnable_levels:

                reg_loss_w = torch.tensor(0, dtype=torch.float32).to(device)
                reg_loss_a = torch.tensor(0, dtype=torch.float32).to(device)

                for i, (names, params) in enumerate(model.named_parameters()):
                    if params.requires_grad and 'bn' not in names:
                        reg_loss_w_current, num_params_w_current = get_closest_loss(params, allowed_weight, reg_sig_w)
                        reg_loss_w += reg_loss_w_current/num_params_w_current

                for activation in activations:
                    reg_loss_a_current, num_params_a_current = get_closest_loss(activation, allowed_weight, reg_sig_a)
                    reg_loss_a += reg_loss_a_current/num_params_a_current

            if args.fixed_levels or args.learnable_levels:
                loss = ce_loss + reg_lambda_a*reg_loss_a + reg_lambda_w*reg_loss_w
            else:
                loss = ce_loss

            loss.backward()
    
            optimizer.step()

            if args.learnable_levels:
                optimizer_levels_w.step()
                optimizer_levels_a.step()

            if args.learnable_levels:
                min_weight = torch.min(allowed_weight)
                max_weight = torch.max(allowed_weight)

                min_tolerance = (allowed_weight[1] - allowed_weight[0])/2
                max_tolerance = (allowed_weight[-1] - allowed_weight[-2])/2

                with torch.no_grad():
                    for (names, params) in model.named_parameters():
                        if params.requires_grad and 'bn' not in names:
                            params.data = torch.clamp(params.data, min_weight - min_tolerance, max_weight + max_tolerance)
            
            if args.fixed_levels:
                with torch.no_grad():
                    for (names, params) in model.named_parameters():
                        if params.requires_grad and 'bn' not in names:
                            params.data = torch.clamp(params.data, allowed_weight[0], allowed_weight[-1])

            running_loss += loss.detach().cpu().item()

            wandb.log({"train_loss": loss.detach().cpu().item()})
            wandb.log({"ce_loss": ce_loss.detach().cpu().item()})

            if args.fixed_levels or args.learnable_levels:
                wandb.log({"reg_loss_w": reg_loss_w.detach().cpu().item()})
                wandb.log({"reg_loss_a": reg_loss_a.detach().cpu().item()})

        scheduler.step()

        if args.learnable_levels:

            scheduler_levels_w.step()
            scheduler_levels_a.step()
            print("allowed weight:", allowed_weight)
            print("allowed act:", allowed_act)


        correct = 0
        total = 0

        model.eval()
        model_copy.eval()

        if epoch % args.epoch_log_test == 0:
            with torch.no_grad():

                if args.fixed_levels:

                    model_copy.load_state_dict(model.state_dict())

                    # clamp weights to allowed levels
                    for (names, params) in model_copy.named_parameters():
                        if params.requires_grad and 'bn' not in names:
                            params.data = torch.clamp(params.data, allowed_weight[0], allowed_weight[-1])
                        
                            for j in range(len(allowed_weight)):
                                params.data = torch.where((params.data >= (allowed_weight[j] - quant_step_w/2)) & (params.data < (allowed_weight[j] + quant_step_w/2)), allowed_weight[j], params.data)

                if args.learnable_levels:

                    # copy all the paramaters of model to model_copy
                    model_copy.load_state_dict(model.state_dict())

                    # sort the allowed weight
                    allowed_weight_copy, _ = torch.sort(allowed_weight)
                    quant_step_array = torch.zeros(len(allowed_weight_copy)-1).to(device)

                    # calculate quant step array from allowed weight
                    for i in range(len(allowed_weight_copy)-1):
                        quant_step_array[i] = allowed_weight_copy[i+1] - allowed_weight_copy[i]

                    # clamp weights to allowed levels
                    for (names, params) in model_copy.named_parameters():
                        if params.requires_grad and 'bn' not in names:
                            params.data = torch.clamp(params.data, allowed_weight_copy[0], allowed_weight_copy[-1])

                            for j in range(len(allowed_weight_copy)):
                                if j == 0:
                                    params.data = torch.where(params.data < (allowed_weight_copy[j] + quant_step_array[j]/2), allowed_weight_copy[j], params.data)
                                elif j == len(allowed_weight_copy)-1:
                                    params.data = torch.where(params.data >= (allowed_weight_copy[j] - quant_step_array[j-1]/2), allowed_weight_copy[j], params.data)
                                else:
                                    params.data = torch.where((params.data >= (allowed_weight_copy[j] - quant_step_array[j-1]/2)) & (params.data < (allowed_weight_copy[j] + quant_step_array[j]/2)), allowed_weight_copy[j], params.data)

                for i, (inputs, labels) in enumerate(testloader, 0):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if args.fixed_levels or args.learnable_levels:
                        output = model_copy.infer(inputs, allowed_act)
                    else:
                        output = model(inputs)

                    # calculate accuracy
                    total += labels.size(0)
                    correct += (output.max(1)[1] == labels).sum().item()

                    top1_acc = top1_acc_compute.update(output, labels)

                top1_acc = top1_acc_compute.compute()
                accuracy_test = 100 * correct / total

                accuracy_test_history.append(accuracy_test)
                max_accuracy = max(accuracy_test_history)

                wandb.log({"test_acc": accuracy_test, "max_test_acc": max_accuracy})
                wandb.log({"top1 acc": top1_acc})

                print('Epoch: %d' % (epoch + 1))
                print('Test accuracy: %.3f %%' % (accuracy_test))
                print('Max test accuracy: %.3f %%' % (max_accuracy))
         

# save both model and model_copy
                
    if args.save_model and not args.learnable_levels and not args.fixed_levels:

        torch.save(model.state_dict(), "./saved_models/" + str(args.dataset) + '_' + str(args.model) + '_' + 'non_quantized' + ".ckpt")
                
    # if args.learnable_levels:
    #     torch.save(model.state_dict(), "./saved_models/vgg13_cifar10_normal_learn_" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")
    #     torch.save(model_copy.state_dict(), "./saved_models/vgg13_cifar10_quantized_learn" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")

    # else:       
    #     torch.save(model.state_dict(), "./saved_models/vgg13_cifar10_normal_" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")
    #     torch.save(model_copy.state_dict(), "./saved_models/vgg13_cifar10_quantized_" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")  


                
if __name__ == '__main__':
    main()