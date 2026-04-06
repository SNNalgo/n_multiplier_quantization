import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from models import *
from utils import *
import resnet_act
import resnet_cifar_act
import wandb
import numpy as np
import matplotlib.pyplot as plt
from make_dataset import *
import time

wandb.init(project="quantization-new")

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
parser.add_argument('--lr_levels_r', default=1e-4, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--lr_levels_a', default=1e-4, type=float, help='learning rate for training the linear classifier')
parser.add_argument('--reg_lambda_w', default=1e-2, type=float, help='lambda for quantization')
parser.add_argument('--reg_lambda_a', default=1e-2, type=float, help='lambda for quantization')

parser.add_argument('--num_bits', default=4, type=int, help='number of bits for quantization')
parser.add_argument('--w_limit', default=0.1, type=float, help='weight limit for quantization')
parser.add_argument('--a_limit', default=5, type=float, help='activation limit for quantization')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--use_pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--fixed_levels', action='store_true', help='use fixed quantization levels')
parser.add_argument('--learnable_levels', action='store_true', help='use learnable quantization levels')
parser.add_argument('--no_act', action='store_true', help='no activation used during quant')
parser.add_argument('--clamp_activations', action='store_true', help='clamp activations')
parser.add_argument('--use_mse_loss', action='store_true', help='use mse loss')
parser.add_argument('--no_activation_quant', action='store_true', help='no activation quantization')
parser.add_argument('--quantize_fl', action='store_true', help='quantize first and last layer')
parser.add_argument('--save_model', action='store_true', help='save model')

args = parser.parse_args()

wandb.config.update(args)

def main():

    torch.manual_seed(args.seed)
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

    num_layers_total = 0
    num_layers_to_quantize = 0

    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
            num_layers_total += 1
    
    print("Number of layers:", num_layers_total)

    if not args.quantize_fl:
        num_layers_to_quantize = num_layers_total - 2
    else:
        num_layers_to_quantize = num_layers_total

    R_matrix = torch.zeros(num_layers_to_quantize, (args.num_bits+1)).to(device)
    reg_sig_w_values = torch.zeros(num_layers_to_quantize).to(device)

    if not args.quantize_fl:

        k = 0
        count = 0

        for (names, params) in model.named_parameters():
            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                if count == 0 or count == num_layers_to_quantize+1:
                    count += 1
                else:
                    average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
                    q_p = (2**args.num_bits - 1)/2
                    scale = 2*average_mag_weights/np.sqrt(q_p)
                    max_limit = q_p*scale
                    reg_sig_w_values[k] = torch.tensor(2*max_limit/(2**args.num_bits-1)).to(device)
                    R_matrix[k] = torch.tensor(get_R_vector(args.num_bits, max_limit)).to(device)
                    print(names)
                    print("reg sig w:", reg_sig_w_values[k])
                    print("average magnitude of weights:", average_mag_weights)
                    print("scale:", scale)
                    print("max value which is q_p*s:", max_limit)
                    count += 1
                    k += 1

    else:
        count = 0
        for (names, params) in model.named_parameters():
            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:

                    average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
                    q_p = (2**args.num_bits - 1)/2
                    scale = 2*average_mag_weights/np.sqrt(q_p)
                    max_limit = q_p*scale
                    print(2*max_limit)
                    print(2**args.num_bits -1)
                    print(2*max_limit/2**args.num_bits-1)
                    reg_sig_w_values[count] = torch.tensor(2*max_limit/(2**args.num_bits-1)).to(device)
                    R_matrix[count] = torch.tensor(get_R_vector(args.num_bits, max_limit)).to(device)
                    print(names)
                    print("reg sig w:", reg_sig_w_values[count])
                    print("average magnitude of weights:", average_mag_weights)
                    print("scale:", scale)
                    print("max value which is q_p*s:", max_limit)
                    count += 1

    S_matrix = torch.tensor(get_S_matrix(args.num_bits), dtype=torch.float32).to(device)
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

    print("Number of params per layer to quantize:", num_params_per_layer_to_quantize)
    num_params_per_layer_to_quantize = np.array(num_params_per_layer_to_quantize)
    num_params_per_layer = np.array(num_params_per_layer)


    max_val_a = 2**args.num_bits - 1
    quant_step_a = np.float32((args.a_limit)/max_val_a)
    reg_sig_a = quant_step_a

    reg_lambda_w = torch.tensor(args.reg_lambda_w).to(device)
    reg_lambda_a = torch.tensor(args.reg_lambda_a).to(device)

    allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)
    allowed_act = torch.tensor(np.arange(0, 2**args.num_bits)*quant_step_a, dtype=torch.float32).to(device)

    if args.learnable_levels:

        R_matrix = nn.Parameter(R_matrix)

        optimizer_levels_r = optim.SGD([R_matrix], lr=args.lr_levels_r, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_levels_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_levels_r, T_max=args.num_epochs)

    if args.learnable_levels and not args.no_act:
 
        allowed_act = nn.Parameter(allowed_act)

        optimizer_levels_a = optim.SGD([allowed_act], lr=args.lr_levels_a, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_levels_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_levels_a, T_max=args.num_epochs)


    print("r values", R_matrix) 
    print("allowed weight:", allowed_weight)

    #print("allowed act:", allowed_act)

    accuracy_test_history = []

    for epoch in range(args.num_epochs): 

        q_p = 2**args.num_bits - 1
        q_p = torch.tensor(q_p).to(device)
        num_bits = torch.tensor(args.num_bits).to(device)
        running_loss = 0.0
        grad_weights_average = torch.tensor(np.zeros((len(trainloader), num_layers_to_quantize))).to(device)
        weights_average = torch.tensor(np.zeros((len(trainloader), num_layers_to_quantize))).to(device)

        grad_r_average = torch.tensor(np.zeros((len(trainloader), num_layers_to_quantize))).to(device)
        r_average = torch.tensor(np.zeros((len(trainloader), num_layers_to_quantize))).to(device)

        model.train()

        for i, (inputs, labels) in enumerate(trainloader, 0):
            allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)
            # print(allowed_weight.size())

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if args.learnable_levels:
                optimizer_levels_r.zero_grad()

            if args.learnable_levels and not args.no_act:
                optimizer_levels_a.zero_grad()

            if args.clamp_activations:
                output, activations = model.forward_clamp(inputs, allowed_act)
            elif (args.fixed_levels or args.learnable_levels) and not args.no_act:
                output, activations = model.forward_act(inputs)
            elif (args.fixed_levels or args.learnable_levels) and args.no_act:
                output = model(inputs)
            else:
                output = model(inputs)

            ce_loss = criterion(output, labels)

            if args.fixed_levels or args.learnable_levels:

                reg_loss_w = torch.tensor(0, dtype=torch.float32).to(device)

                if not args.quantize_fl:
                    count = 0
                    k = 0

                    for (names, params) in model.named_parameters():
                        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                            if count == 0 or count == num_layers_to_quantize+1:
                                count += 1
                            else:
                                if args.use_mse_loss:
                                    reg_loss_current, num_params_current = get_closest_loss_mse(params, allowed_weight[k])
                                    num_params_current = torch.tensor(num_params_current).to(device)
                                    #reg_loss_w += reg_loss_current/(num_params_current*q_p)
                                    reg_loss_w += (num_bits**2)*reg_loss_current/(num_params_current)
                                    count += 1
                                    k += 1
                                else:
                                    reg_loss_current, num_params_current = get_closest_loss(params, allowed_weight[k], reg_sig_w_values[k])
                                    num_params_current = torch.tensor(num_params_current).to(device)
                                    reg_loss_w += reg_loss_current/num_params_current
                                    count += 1
                                    k += 1

                else:
                    count = 0

                    for (names, params) in model.named_parameters():
                        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                            if args.use_mse_loss:
                                reg_loss_current, num_params_current = get_closest_loss_mse(params, allowed_weight[count])
                                num_params_current = torch.tensor(num_params_current).to(device)
                                #reg_loss_w += reg_loss_current/(num_params_current*q_p)
                                reg_loss_w += (num_bits**2)*reg_loss_current/(num_params_current)
                                count += 1
                            else:
                                reg_loss_current, num_params_current = get_closest_loss(params, allowed_weight[count], reg_sig_w_values[count])
                                num_params_current = torch.tensor(num_params_current).to(device)
                                reg_loss_w += reg_loss_current/num_params_current
                                count += 1

            if not args.no_act:

                reg_loss_a = torch.tensor(0, dtype=torch.float32).to(device)

                for activation in activations:
                    reg_loss_current, num_params_current = get_closest_loss(activation, allowed_act, reg_sig_a)
                    reg_loss_a += reg_loss_current/num_params_current

            if (args.fixed_levels or args.learnable_levels) and not args.no_act:
                loss = ce_loss + reg_lambda_a*reg_loss_a + reg_lambda_w*reg_loss_w
            elif (args.fixed_levels or args.learnable_levels) and args.no_act:
                loss = ce_loss + reg_lambda_w*reg_loss_w
            else:
                loss = ce_loss

            loss.backward()
    
            optimizer.step()

            if args.learnable_levels:
                optimizer_levels_r.step()

            if args.learnable_levels and not args.no_act:
                optimizer_levels_a.step()

            if args.learnable_levels and not args.use_mse_loss:
                min_weight = torch.min(allowed_weight, axis=1).values
                max_weight = torch.max(allowed_weight, axis=1).values

                min_tolerance = (allowed_weight[:,1] - allowed_weight[:,0])/2
                max_tolerance = (allowed_weight[:,-1] - allowed_weight[:,-2])/2

                with torch.no_grad():
                    if not args.quantize_fl:
                        count = 0
                        k = 0
                        for (names, params) in model.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                                if count == 0 or count == num_layers_to_quantize+1:
                                    count += 1
                                else:
                                    params.data = torch.clamp(params.data, min_weight[k] - min_tolerance[k], max_weight[k] + max_tolerance[k])
                                    count += 1
                                    k += 1
                    else:
                        count = 0
                        for (names, params) in model.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                                params.data = torch.clamp(params.data, min_weight[count] - min_tolerance[count], max_weight[count] + max_tolerance[count])
                                count += 1
            
            if args.fixed_levels and not args.use_mse_loss:
                
                with torch.no_grad():

                    if not args.quantize_fl:
                        count = 0
                        k = 0
                        for (names, params) in model.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                                if count == 0 or count == num_layers_to_quantize+1:
                                    count += 1
                                else:
                                    params.data = torch.clamp(params.data, allowed_weight[k, 0], allowed_weight[k, -1])
                                    count += 1
                                    k += 1
                    else:
                        count = 0
                        for (names, params) in model.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                                params.data = torch.clamp(params.data, allowed_weight[count, 0], allowed_weight[count, -1])
                                count += 1

            # if we are in the first epoch
            if epoch <= args.num_epochs:
                # calculate ratio of average magnitude of grad of weights to the average magnitude of weights per layer
                grad_weights = []
                weights = []

                count = 0

                if not args.quantize_fl:
                    for (names, params) in model.named_parameters():
                        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                            if count == 0 or count == num_layers_to_quantize+1:
                                count += 1
                            else:
                                # append the norm of grad weights instead of grad weights
                                grad_weights
                                grad_weights.append(torch.mean(torch.abs(params.grad)))
                                weights.append(torch.mean(torch.abs(params)))
                                count += 1
                else:

                    for (names, params) in model.named_parameters():
                        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                            grad_weights.append(torch.mean(torch.abs(params.grad)))
                            weights.append(torch.mean(torch.abs(params)))


                grad_weights = torch.tensor(grad_weights).to(device)
                weights = torch.tensor(weights).to(device)

                # average out grad_weights and weights over all iterations
                grad_weights_average[i, :] = grad_weights
                weights_average[i, :] = weights

                if args.learnable_levels and not args.use_mse_loss:
                    grad_r = []
                    r = []

                    grad_r = torch.mean(torch.abs(R_matrix.grad), axis=1)
                    r = torch.mean(torch.abs(R_matrix), axis=1)

                    grad_r_average[i, :] = grad_r
                    r_average[i, :] = r

            running_loss += loss.detach().cpu().item()

            wandb.log({"train_loss": loss.detach().cpu().item()})
            wandb.log({"ce_loss": ce_loss.detach().cpu().item()})

            if (args.fixed_levels or args.learnable_levels):
                wandb.log({"reg_loss_w": reg_loss_w.detach().cpu().item()})
            
            if (args.fixed_levels or args.learnable_levels) and not args.no_act:
                wandb.log({"reg_loss_a": reg_loss_w.detach().cpu().item()})
        
        if epoch <= args.num_epochs and args.learnable_levels and not args.use_mse_loss:
            # average out grad_weights and weights over all iterations
            grad_weights_average = torch.mean(grad_weights_average, axis=0)
            weights_average = torch.mean(weights_average, axis=0)
            ratios_weights = grad_weights_average/weights_average

            grad_r_average = torch.mean(grad_r_average, axis=0)
            r_average = torch.mean(r_average, axis=0)
            ratios_r = grad_r_average/r_average

            ratios_weights_to_r = ratios_weights/ratios_r
            
            mega_average_grad_weights = torch.mean(grad_weights_average)
            mega_average_weights = torch.mean(weights_average)
            mega_average_ratios_weights = torch.mean(ratios_weights)

            mega_average_grad_r = torch.mean(grad_r_average)
            mega_average_r = torch.mean(r_average)
            mega_average_ratios_r = torch.mean(ratios_r)


            # calculate weughted ratio by number of params per layer of ratios_weights_to_r

            # convert num_params_per_layer to tensor
            num_params_per_layer_to_quantize_tensor = torch.tensor(num_params_per_layer_to_quantize).to(device)
            mega_ratio = torch.sum(ratios_weights_to_r * num_params_per_layer_to_quantize_tensor) / torch.sum(num_params_per_layer_to_quantize_tensor)

            wandb.log({"mega_ratio": mega_ratio})
            wandb.log({"mega_average_grad_weights": mega_average_grad_weights})
            wandb.log({"mega_average_weights": mega_average_weights})
            wandb.log({"mega_average_ratios_weights": mega_average_ratios_weights})
            wandb.log({"mega_average_grad_r": mega_average_grad_r})
            wandb.log({"mega_average_r": mega_average_r})
            wandb.log({"mega_average_ratios_r": mega_average_ratios_r})

            # calculate weighted ratio of 

            print("grad_weights_average:", grad_weights_average)
            print("weights_average:", weights_average)

            print("grad_r_average:", grad_r_average)
            print("r_average:", r_average)

            print("ratios weights:", ratios_weights)
            print("ratios r:", ratios_r)

            print("ratio of weights to r:", ratios_weights_to_r)

            print("mega ratio:", mega_ratio)

            print("number of params per layer:", num_params_per_layer)
            print("sqrt of num params per layer:", np.sqrt(num_params_per_layer))

        print("Epoch:", epoch+1)

        scheduler.step()

        with torch.no_grad():
            allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)

        if args.learnable_levels:

            scheduler_levels_r.step()

            if (epoch+1) % 5 == 0:
                print("r values", R_matrix)
                print("allowed weight:", allowed_weight)


        if args.learnable_levels and not args.no_act:

            scheduler_levels_a.step()

            if (epoch+1) % 5 == 0:
                print("allowed act:", allowed_act)

        correct = 0
        total = 0
        scale_factors_activations = []

        model.eval()
        model_copy.eval()

        if ((epoch + 1) % args.epoch_log_test == 0):
            with torch.no_grad():

                model_copy.load_state_dict(model.state_dict())

                if args.learnable_levels or args.fixed_levels:

                    # sort the allowed weight
                    # allowed_weight_copy, _ = torch.sort(allowed_weight, axis=1)
                    allowed_weight_copy = allowed_weight
                    quant_step_array = torch.zeros(allowed_weight_copy.size()[1]-1).to(device)
                    quant_step_array = quant_step_array.repeat(num_layers_to_quantize, 1)

                    # calculate quant step array from allowed weight
                    for i in range(allowed_weight_copy.size()[1]-1):
                        quant_step_array[:, i] = allowed_weight_copy[:, i+1] - allowed_weight_copy[:, i]
                    
                    # clamp weights to allowed levels
                    if not args.quantize_fl:
                        count = 0
                        k = 0
                        for (names, params) in model_copy.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                                if count == 0 or count == num_layers_to_quantize+1:
                                    count += 1
                                else:
                                    params.data = torch.clamp(params.data, allowed_weight_copy[k, 0], allowed_weight_copy[k, -1])

                                    for j in range(allowed_weight_copy.size()[1]-1):
                                        if j == 0:
                                            params.data = torch.where(params.data < (allowed_weight_copy[k, j] + quant_step_array[k, j]/2), allowed_weight_copy[k, j], params.data)
                                        elif j == allowed_weight_copy.size()[1]-1:
                                            params.data = torch.where(params.data >= (allowed_weight_copy[k, j] - quant_step_array[k, j-1]/2), allowed_weight_copy[k, j], params.data)
                                        else:
                                            params.data = torch.where((params.data >= (allowed_weight_copy[k, j] - quant_step_array[k, j-1]/2)) & (params.data < (allowed_weight_copy[k, j] + quant_step_array[k, j]/2)), allowed_weight_copy[k, j], params.data)
                    
                                    count += 1
                                    k += 1
                    else:                
                        count = 0
                        for (names, params) in model_copy.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names:
                                params.data = torch.clamp(params.data, allowed_weight_copy[count, 0], allowed_weight_copy[count, -1])

                                for j in range(allowed_weight_copy.size()[1]-1):
                                    if j == 0:
                                        params.data = torch.where(params.data < (allowed_weight_copy[count, j] + quant_step_array[count, j]/2), allowed_weight_copy[count, j], params.data)
                                    elif j == allowed_weight_copy.size()[1]-1:
                                        params.data = torch.where(params.data >= (allowed_weight_copy[count, j] - quant_step_array[count, j-1]/2), allowed_weight_copy[count, j], params.data)
                                    else:
                                        params.data = torch.where((params.data >= (allowed_weight_copy[count, j] - quant_step_array[count, j-1]/2)) & (params.data < (allowed_weight_copy[count, j] + quant_step_array[count, j]/2)), allowed_weight_copy[count, j], params.data)
                                
                                count += 1

                for i, (inputs, labels) in enumerate(testloader, 0):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # in the first batch, calculate activation scale factor values
                    if i == 0 and not args.no_activation_quant:
                        output, activations = model_copy.forward_act(inputs)

                        for act in activations:
                            non_zero_act = act[act != 0]
                            average_mag_act = np.mean(np.abs(non_zero_act.detach().cpu().numpy()))
                            q_p = 2**args.num_bits - 1
                            scale_factor = 2*average_mag_act/np.sqrt(q_p)

                            scale_factors_activations.append(scale_factor)

                        print("scale factors activations:", scale_factors_activations)

                    if (args.fixed_levels or args.learnable_levels) and not args.no_activation_quant:
                        output = model_copy.infer_traditional_act(inputs, args.num_bits, scale_factors_activations, quantize_fl=args.quantize_fl)
                    elif (args.fixed_levels or args.learnable_levels) and args.no_activation_quant:
                        output = model_copy(inputs)
                    else:
                        output = model(inputs)

                    # calculate accuracy
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

    if args.save_model and args.learnable_levels and ((epoch+1) % 25 == 0):
        torch.save(model.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_' + 'learnable_fp' + '_' + str(args.num_bits)
                        + "_lambda_" + str(args.reg_lambda_w) + "_lr_" + str(args.lr_levels_r) + "_epoch_" + str(epoch+1) + ".ckpt")
        torch.save(model_copy.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_' + 'learnable_quantized' + '_' + str(args.num_bits)
                          + "_lambda_" + str(args.reg_lambda_w) + "_lr_" + str(args.lr_levels_r) + "_epoch_" + str(epoch+1) + ".ckpt")

    if args.save_model and args.fixed_levels:
        torch.save(model.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_' + 'fixed_fp' + '_' + str(args.num_bits)
                       + "_lambda_" + str(args.reg_lambda_w) + "_lr_" + str(args.lr_levels_r) + ".ckpt")
        torch.save(model_copy.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_' + 'fixed_quantized' + '_' + str(args.num_bits)
                          + "_lambda_" + str(args.reg_lambda_w) + "_lr_" + str(args.lr_levels_r) + ".ckpt")


        # ...
                
    # if args.learnable_levels:
    #     torch.save(model.state_dict(), "./saved_models/vgg13_cifar10_normal_learn_" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")
    #     torch.save(model_copy.state_dict(), "./saved_models/vgg13_cifar10_quantized_learn" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")

    # else:       
    #     torch.save(model.state_dict(), "./saved_models/vgg13_cifar10_normal_" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")
    #     torch.save(model_copy.state_dict(), "./saved_models/vgg13_cifar10_quantized_" + str(args.num_bits) + '_' + str(args.reg_lambda) + ".ckpt")  

                
if __name__ == '__main__':

    main()
