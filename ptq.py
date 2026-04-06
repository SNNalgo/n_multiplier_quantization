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
import numpy as np
import matplotlib.pyplot as plt
from make_dataset import *

parser = argparse.ArgumentParser(description='Quant Aware Training')

parser.add_argument('--batch_size_test', default=256, type=int, help='batch size for training the linear classifier')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading the data')
parser.add_argument('--device', default='cuda', type=str, help='device to be used for training the linear classifier')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to be used for training the linear classifier')
parser.add_argument('--model', default='resnet20', type=str, help='model to be used for training the linear classifier')
parser.add_argument('--seed', default=1, type=int, help='seed for random number generator')
parser.add_argument('--num_bits', default=4, type=int, help='number of bits for quantization')
parser.add_argument('--quant_method', default='traditional', type=str, help='quantization method to be used')
parser.add_argument('--activation_quant', action='store_true', help='quantize activations')

args = parser.parse_args()


def main():

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    trainset, testset = get_dataset(args.dataset)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers, drop_last=False)

    print(len(testset))

    if args.model == 'vgg13':
        model = VGG13_Act(affine=False, bias=True).to(args.device)
        model_copy = VGG13_Act(affine=False, bias=True).to(args.device)

        model.load_state_dict(torch.load("./saved_models/cifar10_vgg13_non_quantized.ckpt"))

        # with non-zero
        #scale_factors_act = [0.351, 0.323, 0.342, 0.403, 0.397, 0.404, 0.401, 0.41, 0.415, 0.476, 0.811, 0.608]
        # with zero also
        scale_factors_act = [0.179, 0.177, 0.191, 0.205, 0.203, 0.204, 0.203, 0.2, 0.199, 0.183, 0.472, 0.451]

    if args.model == 'resnet18':
        model = resnet_act.resnet18().to(args.device)
        model_copy = resnet_act.resnet18().to(args.device)

        model.load_state_dict(torch.load("./saved_models/imagenet_resnet18_non_quantized.ckpt"))

    if args.model == 'resnet20':
        model = resnet_cifar_act.resnet20().to(args.device)
        model_copy = resnet_cifar_act.resnet20().to(args.device)

        model.load_state_dict(torch.load("./saved_models/cifar10_resnet20_non_quantized.ckpt"))

        scale_factors_act = [0.055, 0.054, 0.076, 0.06, 0.096, 0.068, 0.086, 0.064, 0.12, 0.067, 0.103, 0.064, 0.149, 0.069, 0.121, 0.07, 0.181]

    num_layers = 0

    for (names, params) in model.named_parameters():
        if params.requires_grad and 'bn' not in names and 'downsample' not in names:
            num_layers += 1

    print("Number of layers:", num_layers)

    R_matrix = torch.zeros(num_layers, (args.num_bits+1)).to(device)
    scale_values = torch.zeros(num_layers).to(device)

    if args.quant_method == 'traditional':
        count = 0
        for (names, params) in model.named_parameters():
            if params.requires_grad and 'bn' not in names and 'downsample' not in names:
                average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
                q_p = 2**(args.num_bits-1) - 1
                scale = 2*average_mag_weights/np.sqrt(q_p)
                scale_values[count] = scale
                print(names)
                print("average magnitude of weights:", average_mag_weights)
                print("scale:", scale)
                print("max value which is q_p*s:", q_p*scale)
                count += 1
        
        print("scale values:", scale_values)
        print("Number of layers:", num_layers)

    else:

        count = 0
        for (names, params) in model.named_parameters():
            if params.requires_grad and 'bn' not in names and 'downsample' not in names:
                average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
                q_p = (2**args.num_bits - 1)/2
                scale = 2*average_mag_weights/np.sqrt(q_p)
                max_limit = q_p*scale
                R_matrix[count] = torch.tensor(get_R_vector(args.num_bits, max_limit)).to(device)
                print(names)
                print("average magnitude of weights:", average_mag_weights)
                print("scale:", scale)
                print("max value which is q_p*s:", q_p*scale)
                count += 1


        print("Number of layers:", num_layers)

        S_matrix = torch.tensor(get_S_matrix(args.num_bits), dtype=torch.float32).to(device)
        allowed_weight = torch.matmul(R_matrix, S_matrix.T).to(device)

        print("R matrix:", R_matrix)
        print("S matrix:", S_matrix)
        print("Allowed weight matrix:", allowed_weight)

    correct = 0
    total = 0

    model.eval()
    model_copy.eval()

    with torch.no_grad():

        model_copy.load_state_dict(model.state_dict())

        if args.quant_method == 'traditional':
            count = 0

            for (names, params) in model_copy.named_parameters():
                if params.requires_grad and 'bn' not in names and 'downsample' not in names:
                    params.data = quantize_traditional(params.data, args.num_bits, scale_values[count])
                    count += 1

        else:
            allowed_weight_copy = allowed_weight
            quant_step_array = torch.zeros(allowed_weight_copy.size()[1]-1).to(device)
            quant_step_array = quant_step_array.repeat(num_layers, 1)

            # calculate quant step array from allowed weight
            for i in range(allowed_weight_copy.size()[1]-1):
                quant_step_array[:, i] = allowed_weight_copy[:, i+1] - allowed_weight_copy[:, i]
            
            # clamp weights to allowed levels
            count = 0
            for (names, params) in model_copy.named_parameters():
                if params.requires_grad and 'bn' not in names and 'downsample' not in names:
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

            if args.activation_quant:
                output = model_copy.infer_traditional_act(inputs, args.num_bits, scale_factors_act)
            else:
                output = model_copy(inputs)

            # calculate accuracy
            total += labels.size(0)
            correct += (output.max(1)[1] == labels).sum().item()

        accuracy_test = 100 * correct / total

        print('Test accuracy: %.3f %%' % (accuracy_test))

                
if __name__ == '__main__':
    main()