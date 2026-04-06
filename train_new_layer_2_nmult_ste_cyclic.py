import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from models import *
from utils import *
import resnet_nmult_ste
import resnet_cifar_act
import wandb
import numpy as np
import matplotlib.pyplot as plt
from make_dataset import *
import torchmetrics
from torchinfo import summary

wandb.init(project="qat experimentation new layer grad factors")

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--batch_size_train', default=128, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=128, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=120, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--start_lambda_schedule_epoch', default=60, type=int, help='number of epochs after which to start lambda schedule')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading the data')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--epoch_log_test', default=1, type=int, help='number of epochs after which to log test accuracy')

parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--model', default='resnet18', type=str, help='model to be used for training the linear classifier')
parser.add_argument('--seed', default=1, type=int, help='seed for random number generator')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_levels_r', default=1e-6, type=float, help='learning rate for R_vectors (Quantization model)')
parser.add_argument('--levels_opt_steps', default=1000, type=int, help='number of steps to optimize the bit multipliers')
parser.add_argument('--reg_lambda_w', default=100, type=float, help='lambda for quantization')
parser.add_argument('--reg_lambda_w_max', default=1000, type=float, help='max lambda for quantization')

parser.add_argument('--num_bits', default=4, type=int, help='number of bits for quantization')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')

parser.add_argument('--use_pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--fixed_levels', action='store_true', help='use fixed quantization levels')
parser.add_argument('--learnable_levels', action='store_true', help='use learnable quantization levels')
#parser.add_argument('--use_mse_loss', action='store_true', help='use mse loss') # use mse loss by default
parser.add_argument('--use_grad_scale', action='store_true', help='use LSQ grad scale')
#parser.add_argument('--quantize_fl', action='store_true', help='quantize first and last layer') # Quantization of first and last layer not allowed
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

        if args.use_pretrained:
            model.load_state_dict(torch.load("./saved_models/cifar10_vgg13_non_quantized.ckpt"))

    if args.model == 'resnet18':
        #model_copy is no longer required since model has quantization inbuilt with STE
        model = resnet_nmult_ste.resnet18(nbits_a=args.num_bits, nbits=args.num_bits).to(args.device)

        #summary(model)
        if args.use_pretrained:
            pretrained_model_path = "./saved_models/imagenet_resnet18_lsq_act_non_quantized_for_" + str(args.num_bits) + "bits.ckpt"
            model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    if args.model == 'resnet20':
        model = resnet_cifar_act.resnet20().to(args.device)

        if args.use_pretrained:
            model.load_state_dict(torch.load("./saved_models/cifar10_resnet20_non_quantized.ckpt"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    num_layers_total = 0
    num_layers_to_quantize = 0

    print("before first batch execution")
    for (names, buffers) in model.named_buffers():
        #print(names)
        if 'alpha' in names:
            print(names)
            print(buffers)
        #if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names and 'weight' in names:
        #    num_layers_total += 1
    
    model.train()
    inputs, labels = next(iter(trainloader))
    inputs = inputs.to(device)
    output = model(inputs) #First batch to initialize alpha values

    print("after first batch execution")
    for (names, buffers) in model.named_buffers():
        #print(names)
        if 'alpha' in names:
            print(names)
            print(buffers)
    
    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names and 'weight' in names:
            num_layers_total += 1

    print("Number of layers:", num_layers_total)
    num_layers_to_quantize = num_layers_total - 2 #first and last layers not quantized (default)
    print("Number of layers to quantize:", num_layers_to_quantize)

    
    ###################### Wrap this part into a function ######################
    # Initializing the R_vectors and S_matrix
    #R_matrix shape = (num_layers_to_quantize, args.num_bits)
    #S_matrix shape = (2**args.num_bits, args.num_bits)
    R_matrix = []
    S_matrix = None
    k = 0
    for (names, buffers) in model.named_buffers():
        if 'alpha' in names and 'downsample' not in names and 'actq' not in names:
            print(names)
            R_matrix.append(buffers.detach())
            k += 1
        if 'conv' in names and 'S' in names and S_matrix == None:
            S_matrix = buffers.detach()
    R_matrix = torch.stack(R_matrix, dim=0).to(device)
    
    ############################################################################

    if args.learnable_levels:
        R_matrix = nn.Parameter(R_matrix)
        optimizer_levels_r = optim.SGD([R_matrix], lr=args.lr_levels_r, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_levels_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_levels_r, T_max=args.num_epochs)

    ###################### Wrap this part into a function ######################
    # calculate number of params per layer
    num_params_per_layer = []
    num_params_per_layer_to_quantize = []

    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names and 'weight' in names:
            num_params_per_layer.append(params.numel())
    
    num_params_per_layer_to_quantize = num_params_per_layer[1:-1]

    print("Number of params per layer to quantize:", num_params_per_layer_to_quantize)
    num_params_per_layer_to_quantize = np.array(num_params_per_layer_to_quantize)
    num_params_per_layer = np.array(num_params_per_layer)
    ############################################################################

    reg_lambda_w = torch.tensor(args.reg_lambda_w).to(device)
    reg_lambda_mult_factor = torch.tensor(np.float32((args.reg_lambda_w_max/args.reg_lambda_w)**(1.0/(args.num_epochs - args.start_lambda_schedule_epoch)))).to(device)

    allowed_weight = ((torch.matmul(S_matrix, R_matrix.T) - R_matrix[:,-1]).T).to(device)

    print("r values", R_matrix) 
    print("allowed weight:", allowed_weight)

    accuracy_test_history = []
    top5_acc_history = []

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'imagenet':
        num_classes = 1000
    #top5_acc_compute = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)

    for epoch in range(args.num_epochs): 
        q_p = 2**args.num_bits - 1
        q_p = torch.tensor(q_p).to(device)
        if epoch>args.start_lambda_schedule_epoch:
            reg_lambda_w = reg_lambda_w * reg_lambda_mult_factor
        num_bits = torch.tensor(args.num_bits).to(device)
        running_loss = 0.0

        model.train()

        for i in range(args.levels_opt_steps):
            if args.learnable_levels:
                allowed_weight = ((torch.matmul(S_matrix, R_matrix.T) - R_matrix[:,-1]).T).to(device)
                optimizer_levels_r.zero_grad()
                level_opt_loss = torch.tensor(0, dtype=torch.float32).to(device)
                count = 0
                k = 0
                for (names, params) in model.named_parameters():
                    if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names and 'weight' in names:
                        if count == 0 or count == num_layers_to_quantize+1:
                            count += 1
                        else:
                            reg_loss_current, num_params_current = get_closest_loss_mse(params.detach(), allowed_weight[k])
                            num_params_current = torch.tensor(num_params_current).to(device)
                            if args.use_grad_scale and (epoch>args.start_lambda_schedule_epoch):
                                level_opt_loss += reg_loss_current*torch.sqrt(q_p/num_params_current)
                            else:
                                level_opt_loss += reg_loss_current*q_p/(num_params_current)             # currently used
                            count += 1
                            k += 1
                level_loss = reg_lambda_w*level_opt_loss
                level_loss.backward()
                optimizer_levels_r.step()

        for i, (inputs, labels) in enumerate(trainloader, 0):
            # Copy bit multipliers from R_matrix to the model
            k = 0
            for (names, buffers) in model.named_buffers():
                if 'alpha' in names and 'downsample' not in names and 'actq' not in names:
                    buffers.data.copy_(R_matrix[k].detach())
                    k += 1

            #allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)
            allowed_weight = ((torch.matmul(S_matrix, R_matrix.T) - R_matrix[:,-1]).T).to(device)

            
            ####### Testing #######
            #print("values in R_matrix", R_matrix)
            #print("alpha values in model:")
            #for (names, buffers) in model.named_buffers():
            #    if 'alpha' in names and 'downsample' not in names and 'actq' not in names:
            #        print(buffers)
            #######################
            

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear optimizer grad values
            optimizer.zero_grad()
            
            # Run forward pass
            if args.fixed_levels or args.learnable_levels:
                output = model(inputs) #training with quantization
            else:
                output = model(inputs) #No change - this mode is for pure LSQ training
                #output = model.forward_no_quant(inputs)
            # objective loss computation
            ce_loss = criterion(output, labels)

            ###################### Wrap this part into a function ######################
            # QAT regularization loss computation
            if args.fixed_levels or args.learnable_levels:
                reg_loss_w = torch.tensor(0, dtype=torch.float32).to(device)

                count = 0
                k = 0
                for (names, params) in model.named_parameters():
                    if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names and 'weight' in names:
                        if count == 0 or count == num_layers_to_quantize+1:
                            count += 1
                        else:
                            reg_loss_current, num_params_current = get_closest_loss_mse(params, allowed_weight.detach()[k])
                            num_params_current = torch.tensor(num_params_current).to(device)
                            #reg_loss_w += reg_loss_current/(num_params_current*q_p)                # not used
                            #reg_loss_w += (num_bits**2)*reg_loss_current/(num_params_current)      # not used
                            #if args.use_grad_scale and (epoch>args.start_lambda_schedule_epoch):
                            #    reg_loss_w += reg_loss_current*torch.sqrt(q_p/num_params_current)   # LSQ gradient scaling
                            if args.use_grad_scale and (epoch>args.start_lambda_schedule_epoch):
                                reg_loss_w += reg_loss_current*torch.sqrt(q_p/num_params_current)
                            else:
                                reg_loss_w += reg_loss_current*q_p/(num_params_current)             # currently used
                            count += 1
                            k += 1

            if (args.fixed_levels or args.learnable_levels):
                loss = ce_loss + reg_lambda_w*reg_loss_w
            else:
                loss = ce_loss
            ############################################################################
            
            # Backward pass
            loss.backward()
            # Weight update
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            wandb.log({"train_loss": loss.detach().cpu().item()})
            wandb.log({"ce_loss": ce_loss.detach().cpu().item()})

            if (args.fixed_levels or args.learnable_levels):
                wandb.log({"reg_loss_w": reg_loss_w.detach().cpu().item()})

        ########## Don't know what this does (remove) ##########
        if epoch <= args.num_epochs:
            # convert num_params_per_layer to tensor
            num_params_per_layer_to_quantize_tensor = torch.tensor(num_params_per_layer_to_quantize).to(device)
        ########################################################

        print("Epoch:", epoch+1)
        scheduler.step()

        with torch.no_grad():
            # Copy bit multipliers from R_matrix to the model
            k = 0
            for (names, buffers) in model.named_buffers():
                if 'alpha' in names and 'downsample' not in names and 'actq' not in names:
                    buffers.data.copy_(R_matrix[k].detach())
                    k += 1
            # Update allowed_weight
            allowed_weight = ((torch.matmul(S_matrix, R_matrix.T) - R_matrix[:,-1]).T).to(device)
        
        if args.learnable_levels:
            scheduler_levels_r.step()
            if (epoch+1) % 1 == 0:
                print("values in R_matrix", R_matrix)
                print("alpha values in model:")
                for (names, buffers) in model.named_buffers():
                    if 'alpha' in names and 'downsample' not in names and 'actq' not in names:
                        print(buffers)
                print("allowed weight:", allowed_weight)

        correct = 0
        total = 0

        model.eval() # model_copy is no longer required since model has quantization inbuilt with STE

        if ((epoch + 1) % args.epoch_log_test == 0):
            with torch.no_grad():
                top5_acc_compute = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
                for i, (inputs, labels) in enumerate(testloader, 0):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if (args.fixed_levels or args.learnable_levels):
                        output = model(inputs)
                    else:
                        output = model(inputs)
                        #output = model.forward_no_quant(inputs)

                    # calculate accuracy
                    total += labels.size(0)
                    correct += (output.max(1)[1] == labels).sum().item()
                    top5_acc_compute.update(output, labels)

                accuracy_test = 100 * correct / total
                top5_acc = top5_acc_compute.compute()*100

                top5_acc_history.append(top5_acc)
                max_top5_acc = max(top5_acc_history)

                accuracy_test_history.append(accuracy_test)
                max_accuracy = max(accuracy_test_history)

                wandb.log({"test_acc": accuracy_test, "max_test_acc": max_accuracy})
                wandb.log({"top5_acc": top5_acc, "max_top5_acc": max_top5_acc})

                #print('Epoch: %d' % (epoch + 1))
                print('Test accuracy: %.3f %%' % (accuracy_test))
                print('Max test accuracy: %.3f %%' % (max_accuracy))
                print('Max top5 test accuracy: %.3f %%' % (max_top5_acc))
                
                # if args.dataset == 'imagenet':
                #     if (epoch<45 and epoch%5==4) or (epoch>=45 and epoch%4==3):
                #         model.load_state_dict(model_copy.state_dict())

    
    
    R_matrix_np = R_matrix.detach().cpu().numpy()
    S_matrix_np = S_matrix.detach().cpu().numpy()

    if args.save_model and not args.learnable_levels and not args.fixed_levels:
        torch.save(model.state_dict(), "./saved_models/" + str(args.dataset) + '_' + str(args.model) + '_lsq_act_non_quantized_for_' + str(args.num_bits) + "bits.ckpt")
    
    if args.save_model and (args.learnable_levels or args.fixed_levels):
        print('R_matrix: ', R_matrix)
        print('S_matrix: ', S_matrix)
        if args.use_grad_scale:
            torch.save(model.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_nmult_ste_grad_scale_' + str(args.num_bits) + "bits.ckpt")
            np.savez("./new_models/" + str(args.dataset) + '_' + str(args.model) + '_nmult_ste_grad_scale_' + str(args.num_bits) + "bits_RS_mats.npz", R_matrix=R_matrix_np, S_matrix=S_matrix_np)
        else:
            torch.save(model.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_nmult_ste_' + str(args.num_bits) + "bits.ckpt")
            np.savez("./new_models/" + str(args.dataset) + '_' + str(args.model) + '_nmult_ste_' + str(args.num_bits) + "bits_RS_mats.npz", R_matrix=R_matrix_np, S_matrix=S_matrix_np)

if __name__ == '__main__':

    main()
