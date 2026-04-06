import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from models import *
from snn_models import *
from utils import *
import resnet_act
import resnet_cifar_act
import wandb
import numpy as np
import matplotlib.pyplot as plt
from make_dataset import *

wandb.init(project="qat snn")

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--batch_size_train', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading the data')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--epoch_log_test', default=1, type=int, help='number of epochs after which to log test accuracy')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--model', default='resnet20', type=str, help='model to be used for training the linear classifier')
parser.add_argument('--seed', default=1, type=int, help='seed for random number generator')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--lr_levels_r', default=1e-3, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--reg_lambda_w', default=100, type=float, help='lambda for quantization')
parser.add_argument('--num_bits', default=4, type=int, help='number of bits for quantization')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--use_pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--fixed_levels', action='store_true', help='use fixed quantization levels')
parser.add_argument('--learnable_levels', action='store_true', help='use learnable quantization levels')
parser.add_argument('--quantize_fl', action='store_true', help='quantize first and last layer')
parser.add_argument('--save_model', action='store_true', help='save model')

args = parser.parse_args()

wandb.config.update(args)

def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    trainset, testset = get_dataset(args.dataset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers, drop_last=False)

    print(len(trainset))
    print(len(testset))

    if args.model == 'vgg13':
        model = VGG13_Act(affine=False, bias=True).to(args.device)
        model_copy = VGG13_Act(affine=False, bias=True).to(args.device)

        if args.use_pretrained:
            model.load_state_dict(torch.load("./saved_models/cifar10_vgg13_non_quantized.ckpt"))

    if args.model == 'resnet18':
        model = resnet_act.resnet18().to(args.device)
        model_copy = resnet_act.resnet18().to(args.device)

        if args.use_pretrained:
            model.load_state_dict(torch.load("./saved_models/imagenet_resnet18_non_quantized.ckpt"))

    if args.model == 'resnet20':
        model = resnet_cifar_act.resnet20().to(args.device)
        model_copy = resnet_cifar_act.resnet20().to(args.device)

        if args.use_pretrained:
            model.load_state_dict(torch.load("./saved_models/cifar10_resnet20_non_quantized.ckpt"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # calculate number of layers
    num_layers_total = 0
    num_layers_to_quantize = 0

    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
            num_layers_total += 1

    if not args.quantize_fl:
        num_layers_to_quantize = num_layers_total - 2
    else:
        num_layers_to_quantize = num_layers_total

    print("Total Number of layers:", num_layers_total)
    print("Number of layers to quantize:", num_layers_to_quantize)


    # calculate number of params per layer
    num_params_per_layer = []
    num_params_per_layer_to_quantize = []

    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
            num_params_per_layer.append(params.numel())
    
    if not args.quantize_fl:
        num_params_per_layer_to_quantize = num_params_per_layer[1:-1]
    else:
        num_params_per_layer_to_quantize = num_params_per_layer

    num_params_per_layer_to_quantize = np.array(num_params_per_layer_to_quantize)
    num_params_per_layer = np.array(num_params_per_layer)

    print("Number of params per layer to quantize:", num_params_per_layer_to_quantize)

    # calculate R and S matrix for the weights
    # note: for weights the allowed_weights contains layers according to quantize_fl
    R_matrix_weights = torch.zeros(num_layers_to_quantize, (args.num_bits+1)).to(device)
    S_matrix_weights = torch.tensor(get_S_matrix(args.num_bits), dtype=torch.float32).to(device)

    for (names, params) in model.named_parameters():
            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                if count == 0 or count == num_layers_to_quantize+1:
                    count += 1
                else:
                    average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
                    q_p = (2**args.num_bits - 1)/2
                    step_size = 2*average_mag_weights/np.sqrt(q_p)
                    max_limit = q_p*step_size
                    R_matrix_weights[k] = torch.tensor(get_R_vector(args.num_bits, max_limit)).to(device)
                    print(names)
                    print("average magnitude of weights:", average_mag_weights)
                    print("step size:", step_size)
                    print("max value which is q_p*s:", max_limit)
                    count += 1
                    k += 1

    reg_lambda_w = torch.tensor(args.reg_lambda_w).to(device)
    allowed_weight = torch.matmul(R_matrix_weights, S_matrix_weights.T).to(device)

    if args.learnable_levels:

        R_matrix_weights = nn.Parameter(R_matrix_weights)
        optimizer_levels_r = optim.SGD([R_matrix_weights], lr=args.lr_levels_r, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_levels_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_levels_r, T_max=args.num_epochs)

    print("R weight values", R_matrix_weights) 
    print("allowed weight:", allowed_weight)

    accuracy_test_history = []

    for epoch in range(args.num_epochs): 

        qp_w = torch.tensor((2**args.num_bits - 1)/2).to(device)
        num_bits = torch.tensor(args.num_bits).to(device)

        model.train()

        for i, (inputs, labels) in enumerate(trainloader, 0):
            allowed_weight = torch.matmul(R_matrix_weights, S_matrix_weights.T).to(device)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if args.learnable_levels:
                optimizer_levels_r.zero_grad()
            
            output = model(inputs)

            ce_loss = criterion(output, labels)
            
            # calculate reg_loss_w
            if args.fixed_levels or args.learnable_levels:

                reg_loss_w = torch.tensor(0, dtype=torch.float32).to(device)

                count = 0
                k = 0
                for (names, params) in model.named_parameters():
                    if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                        if count == 0 or count == num_layers_total-1:
                            count += 1
                        else:
                            reg_loss_current, num_params_current = get_closest_loss_mse(params, allowed_weight[k])
                            num_params_current = torch.tensor(num_params_current).to(device)
                            reg_loss_w += reg_loss_current*qp_w/(num_params_current)
                            count += 1
                            k += 1
            
            loss = ce_loss + reg_lambda_w*reg_loss_w

            wandb.log({"train_loss": loss.detach().cpu().item()})
            wandb.log({"ce_loss": ce_loss.detach().cpu().item()})
            wandb.log({"reg_loss_w": reg_loss_w.detach().cpu().item()})

            loss.backward()
    
            optimizer.step()

            if args.learnable_levels:
                optimizer_levels_r.step()

            wandb.log({"train_loss": loss.detach().cpu().item()})
            wandb.log({"ce_loss": ce_loss.detach().cpu().item()})

            if (args.fixed_levels or args.learnable_levels):
                wandb.log({"reg_loss_w": reg_loss_w.detach().cpu().item()})
            
            if (args.fixed_levels or args.learnable_levels) and not args.no_act:
                wandb.log({"reg_loss_a": reg_loss_w.detach().cpu().item()})

        print("Epoch:", epoch+1)

        scheduler.step()

        with torch.no_grad():
            allowed_weight = torch.matmul(R_matrix_weights, S_matrix_weights.T).to(device)

        if args.learnable_levels:

            scheduler_levels_r.step()

            if (epoch+1) % 1 == 0:
                print("R weight values", R_matrix_weights)
                print("allowed weight:", allowed_weight)

        correct = 0
        total = 0

        model.eval()
        model_copy.eval()

        if ((epoch + 1) % args.epoch_log_test == 0):
            with torch.no_grad():

                model_copy.load_state_dict(model.state_dict())

                if args.learnable_levels or args.fixed_levels:

                    allowed_weight_copy = allowed_weight.clone()
                
                    count = 0
                    k = 0
                    for (names, params) in model_copy.named_parameters():
                        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                            if count == 0 or count == num_layers_to_quantize+1:
                                count += 1
                            else:
                                params.data = quantize_general(params.data, allowed_weight_copy[k], device)                    
                                count += 1
                                k += 1

                for i, (inputs, labels) in enumerate(testloader, 0):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if args.fixed_levels or args.learnable_level:
                        output = model_copy(inputs)
                    else:
                        output = model(inputs)

                    total += labels.size(0)
                    correct += (output.max(1)[1] == labels).sum().item()

                accuracy_test = 100 * correct / total

                accuracy_test_history.append(accuracy_test)
                max_accuracy = max(accuracy_test_history)

                wandb.log({"test_acc": accuracy_test, "max_test_acc": max_accuracy})

                print('Epoch: %d' % (epoch + 1))
                print('Test accuracy: %.3f %%' % (accuracy_test))
                print('Max test accuracy: %.3f %%' % (max_accuracy))

                
    if args.save_model and not args.learnable_levels and not args.fixed_levels:
        torch.save(model.state_dict(), "./saved_models/" + str(args.dataset) + '_' + str(args.model) + '_' + 'non_quantized' + ".ckpt")
                
            
if __name__ == '__main__':

    main()