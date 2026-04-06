import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse 
from models import *
from utils_var import *
import resnet_lsq
import resnet_cifar_lsq
import wandb
import numpy as np
import matplotlib.pyplot as plt
from make_dataset import *
import torchmetrics

wandb.init(project="qat experimentation new layer grad factors")

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--batch_size_train', default=128, type=int, help='batch size for training the linear classifier')
parser.add_argument('--batch_size_test', default=128, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_epochs', default=120, type=int, help='number of epochs for training the linear classifier')
parser.add_argument('--start_lambda_schedule_epoch', default=100, type=int, help='number of epochs after which to start lambda schedule')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading the data')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for training the linear classifier')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--epoch_log_test', default=1, type=int, help='number of epochs after which to log test accuracy')
parser.add_argument('--wt_load_freq1', default=5, type=int, help='weight load frequency in first half')
parser.add_argument('--wt_load_freq2', default=4, type=int, help='weight load frequency in second half')

parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--model', default='resnet18', type=str, help='model to be used for training the linear classifier')
parser.add_argument('--seed', default=1, type=int, help='seed for random number generator')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
#parser.add_argument('--lr_levels_r', default=1e-6, type=float, help='learning rate for R_vectors (Quantization model)')
parser.add_argument('--reg_lambda_w', default=100, type=float, help='lambda for quantization')
parser.add_argument('--reg_lambda_w_max', default=2000, type=float, help='max lambda for quantization')

parser.add_argument('--num_bits', default=4, type=int, help='number of bits for quantization')
parser.add_argument('--var_sig_by_mu', default=0.02, type=float, help='sigma by mu of LRS ("1" bit)')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')

parser.add_argument('--use_pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--var_aware', action='store_true', help='train with variability awareness')
#parser.add_argument('--learnable_levels', action='store_true', help='use learnable quantization levels')
#parser.add_argument('--use_mse_loss', action='store_true', help='use mse loss') # use mse loss by default
parser.add_argument('--use_grad_scale', action='store_true', help='use LSQ grad scale')
parser.add_argument('--quantize_fl', action='store_true', help='quantize first and last layer')
#parser.add_argument('--fault_aware', action='store_true', help='use fault-aware loss for training')
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

    if args.model == 'resnet20':
        model = resnet_cifar_lsq.resnet20().to(args.device)
        model_copy = resnet_cifar_lsq.resnet20().to(args.device)
        model.load_state_dict(torch.load("./saved_models/cifar10_resnet20_non_quantized.ckpt"), strict=False)

    if args.model == 'resnet18':
        model = resnet_lsq.resnet18(nbits_a=args.num_bits).to(args.device)
        model_copy = resnet_lsq.resnet18(nbits_a=args.num_bits).to(args.device)

        if args.use_pretrained:
            pretrained_model_path = "./new_models/imagenet_resnet18_lsq_act_grad_scale_learnable_quantized_lambda_schedule_" + str(args.num_bits) + "bits.ckpt"
            RS_matrix_path = "./new_models/imagenet_resnet18_lsq_act_grad_scale_learnable_quantized_lambda_schedule_" + str(args.num_bits) + "bits_RS_mats.npz"

            model.load_state_dict(torch.load(pretrained_model_path))
            npzfile = np.load(RS_matrix_path)
            R_matrix_np = npzfile["R_matrix"]
            S_matrix_np = npzfile["S_matrix"]

    if args.model == 'resnet50':
        model, _ = resnet_lsq.resnet50(nbits_a=args.num_bits)
        model_copy, _ = resnet_lsq.resnet50(nbits_a=args.num_bits)
        model = model.to(args.device)
        model_copy = model_copy.to(args.device)

        if args.use_pretrained:
            pretrained_model_path = "./new_models/imagenet_resnet50_lsq_act_grad_scale_learnable_quantized_lambda_schedule_" + str(args.num_bits) + "bits.ckpt"
            RS_matrix_path = "./new_models/imagenet_resnet50_lsq_act_grad_scale_learnable_quantized_lambda_schedule_" + str(args.num_bits) + "bits_RS_mats.npz"

            model.load_state_dict(torch.load(pretrained_model_path))
            npzfile = np.load(RS_matrix_path)
            R_matrix_np = npzfile["R_matrix"]
            S_matrix_np = npzfile["S_matrix"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    num_layers_total = 0
    num_layers_to_quantize = 0

    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
            num_layers_total += 1
    
    print("Number of layers:", num_layers_total)

    if not args.quantize_fl:
        num_layers_to_quantize = num_layers_total - 2
    else:
        num_layers_to_quantize = num_layers_total
    print("Number of layers to quantize:", num_layers_to_quantize)

    params_lvl_valid_list = []
    params_bit_var_list = []
    allowed_weight = []
    ###################### Wrap this part into a function ######################
    # Initializing the R_vectors and S_matrix
    if args.use_pretrained:
        R_matrix = torch.from_numpy(np.float32(R_matrix_np)).to(device)
        S_matrix = torch.from_numpy(np.float32(S_matrix_np)).to(device)
        print('r matrix: ', R_matrix)
    else:
        S_matrix = torch.tensor(get_S_matrix(args.num_bits), dtype=torch.float32).to(device)
        R_matrix = torch.zeros(num_layers_to_quantize, (args.num_bits+1)).to(device)
    
    if not args.quantize_fl: #first and last layers not quantized (default)
        k = 0
        count = 0
        for (names, params) in model.named_parameters():
            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
                if count == 0 or count == num_layers_to_quantize+1:
                    count += 1
                else:
                    # Set up bit variability #
                    params_np_shape = params.detach().cpu().numpy().shape
                    params_bit_var_shape = params_np_shape + (args.num_bits,)
                    params_bit_var_shape_full = params_np_shape + (args.num_bits+1,)
                    params_bit_vars = 1.0 + (args.var_sig_by_mu * np.random.normal(size=params_bit_var_shape))
                    params_bit_vars_full = np.ones(params_bit_var_shape_full)
                    params_bit_vars_full[..., :-1] = params_bit_vars
                    
                    params_bit_vars_t = torch.from_numpy(np.float32(params_bit_vars_full)).to(device)
                    params_bit_var_list.append(params_bit_vars_t)
                    ###################################
                    
                    average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
                    q_p = (2**args.num_bits - 1)/2
                    scale = 2*average_mag_weights/np.sqrt(q_p)
                    max_limit = q_p*scale
                    if not args.use_pretrained:
                        R_matrix[k] = torch.tensor(get_R_vector(args.num_bits, max_limit)).to(device)
                        print(names)
                        print("max value which is q_p*s:", max_limit)
                    R_vector_var = params_bit_vars_t * R_matrix[k]
                    #orig_allowed_wts = torch.matmul(R_matrix[k], S_matrix.T)
                    #level_allowed_wt = torch.matmul(R_vector_var, S_matrix.T)
                    #print('orig_allowed_wts:         ', orig_allowed_wts)
                    #print('example level_allowed_wt: ', level_allowed_wt[0,0,0,0,:])
                    allowed_weight.append(torch.matmul(R_vector_var, S_matrix.T).to(device))
                    
                    count += 1
                    k += 1

    else:
        count = 0
        for (names, params) in model.named_parameters():
            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
                # Set up bit variability #
                params_np_shape = params.detach().cpu().numpy().shape
                params_bit_var_shape = params_np_shape + (args.num_bits,)
                params_bit_var_shape_full = params_np_shape + (args.num_bits+1,)
                params_bit_vars = 1.0 + (args.var_sig_by_mu * np.random.normal(size=params_bit_var_shape))
                params_bit_vars_full = np.ones(params_bit_var_shape_full)
                params_bit_vars_full[..., :-1] = params_bit_vars
                
                params_bit_vars_t = torch.from_numpy(np.float32(params_bit_vars_full)).to(device)
                params_bit_var_list.append(params_bit_vars_t)
                ###################################
                
                average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
                q_p = (2**args.num_bits - 1)/2
                scale = 2*average_mag_weights/np.sqrt(q_p)
                max_limit = q_p*scale
                if not args.use_pretrained:
                    R_matrix[k] = torch.tensor(get_R_vector(args.num_bits, max_limit)).to(device)
                    print(names)
                    print("max value which is q_p*s:", max_limit)
                R_vector_var = params_bit_vars_t * R_matrix[k]
                allowed_weight.append(torch.matmul(R_vector_var, S_matrix.T).to(device))
                
                count += 1
    
    ############################################################################

    ###################### Wrap this part into a function ######################
    # calculate number of params per layer
    num_params_per_layer = []
    num_params_per_layer_to_quantize = []

    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
            num_params_per_layer.append(params.numel())
    
    if not args.quantize_fl:
        num_params_per_layer_to_quantize = num_params_per_layer[1:-1]
    else:
        num_params_per_layer_to_quantize = num_params_per_layer

    print("Number of params per layer to quantize:", num_params_per_layer_to_quantize)
    num_params_per_layer_to_quantize = np.array(num_params_per_layer_to_quantize)
    num_params_per_layer = np.array(num_params_per_layer)
    ############################################################################

    reg_lambda_w = torch.tensor(args.reg_lambda_w).to(device)
    start_lambda_schedule_epoch = args.start_lambda_schedule_epoch
    if start_lambda_schedule_epoch >= args.num_epochs:
        start_lambda_schedule_epoch = args.num_epochs//2
    reg_lambda_mult_factor = torch.tensor(np.float32((args.reg_lambda_w_max/args.reg_lambda_w)**(1.0/(args.num_epochs - start_lambda_schedule_epoch)))).to(device)

    #allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)
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
        if epoch>start_lambda_schedule_epoch:
            reg_lambda_w = reg_lambda_w * reg_lambda_mult_factor
        num_bits = torch.tensor(args.num_bits).to(device)
        running_loss = 0.0
        
        model.train()

        if epoch>0:
            for i, (inputs, labels) in enumerate(trainloader, 0):
                #NOT HANDLING learnable_levels, USE WITH var_aware instead
                #allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)


                inputs = inputs.to(device)
                labels = labels.to(device)

                # Clear optimizer grad values
                optimizer.zero_grad()

                # Run forward pass
                if args.var_aware:
                    output = model(inputs) #training with variability aware quantization
                else:
                    output = model(inputs) #training without variability awareness
                # objective loss computation
                ce_loss = criterion(output, labels)

                ###################### Wrap this part into a function ######################
                # QAT regularization loss computation
                if args.var_aware:
                    reg_loss_w = torch.tensor(0, dtype=torch.float32).to(device)

                    if not args.quantize_fl:
                        count = 0
                        k = 0
                        for (names, params) in model.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
                                if count == 0 or count == num_layers_to_quantize+1:
                                    count += 1
                                else:
                                    reg_loss_current, num_params_current = get_closest_loss_mse(params, allowed_weight[k])
                                    num_params_current = torch.tensor(num_params_current).to(device)
                                    if args.use_grad_scale and epoch>start_lambda_schedule_epoch:
                                        reg_loss_w += reg_loss_current*torch.sqrt(q_p/num_params_current)   # LSQ gradient scaling
                                    else:
                                        reg_loss_w += reg_loss_current*q_p/(num_params_current)             # currently used
                                    count += 1
                                    k += 1
                    else:
                        count = 0
                        for (names, params) in model.named_parameters():
                            if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
                                reg_loss_current, num_params_current = get_closest_loss_mse(params, allowed_weight[count])
                                num_params_current = torch.tensor(num_params_current).to(device)
                                if args.use_grad_scale and epoch>start_lambda_schedule_epoch:
                                    reg_loss_w += reg_loss_current*torch.sqrt(q_p/num_params_current)   # LSQ gradient scaling
                                else:
                                    reg_loss_w += reg_loss_current*q_p/(num_params_current)             # currently used
                                count += 1

                if args.var_aware:
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

                if (args.var_aware):
                    wandb.log({"reg_loss_w": reg_loss_w.detach().cpu().item()})

        ########## Don't know what this does (remove) ##########
        if epoch <= args.num_epochs:
            # convert num_params_per_layer to tensor
            num_params_per_layer_to_quantize_tensor = torch.tensor(num_params_per_layer_to_quantize).to(device)
        ########################################################

        scheduler.step()
        
        print("Epoch:", epoch+1)

        #with torch.no_grad():
        #    allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)

        correct = 0
        total = 0

        model.eval()
        model_copy.eval()

        if ((epoch + 1) % args.epoch_log_test == 0):
            # Quantize weights in model_copy
            with torch.no_grad():
                ########## Wrap this part in a function ##########
                model_copy.load_state_dict(model.state_dict())
                allowed_weight_copy = allowed_weight
                if not args.quantize_fl:
                    count = 0
                    k = 0
                    for (names, params) in model_copy.named_parameters():
                        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
                            if count == 0 or count == num_layers_to_quantize+1:
                                count += 1
                            else:
                                param_copy = params.data + 0
                                min_diff = param_copy*0 + torch.max(param_copy)
                                new_param = param_copy*0
                                for j in range(2**args.num_bits):
                                    abs_diff = torch.abs(param_copy - allowed_weight_copy[k][..., j])
                                    #Set new_param value to "allowed_weight[k][...,j]" if current "abs_diff" is less than "min_diff"
                                    new_param = torch.where(abs_diff<min_diff, allowed_weight_copy[k][..., j], new_param)
                                    #Update "min_level_diff" values if current "abs_level_diff" is less than "min_level_diff"
                                    min_diff = torch.where(abs_diff<min_diff, abs_diff, min_diff)
                                count += 1
                                k += 1
                                params.data = new_param + 0
                else:
                    count = 0
                    for (names, params) in model_copy.named_parameters():
                        if params.requires_grad and 'bn' not in names and 'downsample' not in names and 'bias' not in names and 'actq' not in names:
                            param_copy = params.data + 0
                            min_diff = param_copy*0 + torch.max(param_copy)
                            new_param = param_copy*0
                            for j in range(2**args.num_bits):
                                abs_diff = torch.abs(param_copy - allowed_weight_copy[count][..., j])
                                #Set new_param value to "allowed_weight[k][...,j]" if current "abs_diff" is less than "min_diff"
                                new_param = torch.where(abs_diff<min_diff, allowed_weight_copy[count][..., j], new_param)
                                #Update "min_level_diff" values if current "abs_level_diff" is less than "min_level_diff"
                                min_diff = torch.where(abs_diff<min_diff, abs_diff, min_diff)
                            count += 1
                            params.data = new_param + 0
                ##################################################

                top5_acc_compute = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
                for i, (inputs, labels) in enumerate(testloader, 0):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # Always perform model_copy.forward irrespective of var_aware or not
                    output = model_copy(inputs)

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

                print('Epoch: %d' % (epoch + 1))
                print('Test accuracy: %.3f %%' % (accuracy_test))
                print('Max test accuracy: %.3f %%' % (max_accuracy))
                print('Max top5 test accuracy: %.3f %%' % (max_top5_acc))
                
                #Without args.var_aware, this implements basic kind of variability awareness
                if epoch==0 or (epoch<args.num_epochs/2 and epoch%args.wt_load_freq1==(args.wt_load_freq1 - 1)) or (epoch>=args.num_epochs/2 and epoch%args.wt_load_freq2==(args.wt_load_freq2 - 1)):
                    model.load_state_dict(model_copy.state_dict())

    if args.save_model and args.var_aware:
        if args.use_grad_scale:
            torch.save(model_copy.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_lsq_act_grad_scale_var_aware_' + str(args.num_bits) + "bits.ckpt")
        else:
            torch.save(model_copy.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_lsq_act_var_aware_' + str(args.num_bits) + "bits.ckpt")
    
    if args.save_model and not args.var_aware:
        if args.use_grad_scale:
            torch.save(model_copy.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_lsq_act_grad_scale_no_var_aware_' + str(args.num_bits) + "bits.ckpt")
        else:
            torch.save(model_copy.state_dict(), "./new_models/" + str(args.dataset) + '_' + str(args.model) + '_lsq_act_no_var_aware_' + str(args.num_bits) + "bits.ckpt")

                
if __name__ == '__main__':

    main()
