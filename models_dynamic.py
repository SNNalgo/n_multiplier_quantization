import torch
import torch.nn as nn
from utils import *

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
class VGG13(nn.Module):
    def __init__(self, num_classes=10, affine=True, bias=True):

        super(VGG13, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=bias)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class VGG13_Act(nn.Module):
    def __init__(self, num_classes=10, affine=True, bias=True):
        super(VGG13_Act, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=bias)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
    
    def forward_act(self, x):
        activations = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x)
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x)
        return x, activations
    
    def forward_clamp(self, x, allowed_act):
        activations = []

        max_act = torch.max(allowed_act)
        min_act = torch.min(allowed_act)
        max_tolerance_a = (allowed_act[-1] - allowed_act[-2])/2

        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = torch.clamp(x, min_act, max_act + max_tolerance_a)
                activations.append(x)

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = torch.clamp(x, min_act, max_act + max_tolerance_a)
                activations.append(x)

        return x, activations
    
    def infer(self, x, allowed_act):
        
        quant_step_array = torch.zeros(len(allowed_act)-1)

        # calculate quant step array from allowed weight
        for i in range(len(allowed_act)-1):
            quant_step_array[i] = allowed_act[i+1] - allowed_act[i]

        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):

                x = torch.clamp(x, allowed_act[0], allowed_act[-1])

                for j in range(len(allowed_act)):
                    if j == 0:
                        x = torch.where(x< (allowed_act[j] + quant_step_array[j]/2), allowed_act[j], x)
                    elif j == len(allowed_act)-1:
                        x = torch.where(x >= (allowed_act[j] - quant_step_array[j-1]/2), allowed_act[j], x)
                    else:
                        x = torch.where((x >= (allowed_act[j] - quant_step_array[j-1]/2)) & (x < (allowed_act[j] + quant_step_array[j]/2)), allowed_act[j], x)

        for layer in self.classifier:
            x = layer(x)
            
            if isinstance(layer, nn.ReLU):

                x = torch.clamp(x, allowed_act[0], allowed_act[-1])

                for j in range(len(allowed_act)):
                    if j == 0:
                        x = torch.where(x< (allowed_act[j] + quant_step_array[j]/2), allowed_act[j], x)
                    elif j == len(allowed_act)-1:
                        x = torch.where(x >= (allowed_act[j] - quant_step_array[j-1]/2), allowed_act[j], x)
                    else:
                        x = torch.where((x >= (allowed_act[j] - quant_step_array[j-1]/2)) & (x < (allowed_act[j] + quant_step_array[j]/2)), allowed_act[j], x)

        return x
    
    def infer_traditional_act(self, x, num_bits, scale_values):

        count = 0

        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = quantize_traditional(x, num_bits, scale_values[count], signed=False)
                count += 1

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = quantize_traditional(x, num_bits, scale_values[count], signed=False)
                count += 1

        return x
        

class VGG13_Act_Dynamic(nn.Module):
    def __init__(self, num_classes=10, affine=True, bias=True):
        super(VGG13_Act_Dynamic, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=bias)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def forward_act(self, x):
        activations = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x)
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x)
        return x, activations

    def infer(self, x, allowed_acts):
        cnt = 0
        quant_step_array = torch.zeros(len(allowed_acts[0])-1)
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                allowed_act = allowed_acts[cnt]
                cnt = cnt + 1
                # calculate quant step array from allowed weight
                for i in range(len(allowed_act)-1):
                    quant_step_array[i] = allowed_act[i+1] - allowed_act[i]
                x = torch.clamp(x, allowed_act[0], allowed_act[-1])
                for j in range(len(allowed_act)):
                    if j == 0:
                        x = torch.where(x< (allowed_act[j] + quant_step_array[j]/2), allowed_act[j], x)
                    elif j == len(allowed_act)-1:
                        x = torch.where(x >= (allowed_act[j] - quant_step_array[j-1]/2), allowed_act[j], x)
                    else:
                        x = torch.where((x >= (allowed_act[j] - quant_step_array[j-1]/2)) & (x < (allowed_act[j] + quant_step_array[j]/2)), allowed_act[j], x)

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                allowed_act = allowed_acts[cnt]
                cnt = cnt + 1
                quant_step_array = torch.zeros(len(allowed_act)-1)
                # calculate quant step array from allowed weight
                for i in range(len(allowed_act)-1):
                    quant_step_array[i] = allowed_act[i+1] - allowed_act[i]
                x = torch.clamp(x, allowed_act[0], allowed_act[-1])
                for j in range(len(allowed_act)):
                    if j == 0:
                        x = torch.where(x< (allowed_act[j] + quant_step_array[j]/2), allowed_act[j], x)
                    elif j == len(allowed_act)-1:
                        x = torch.where(x >= (allowed_act[j] - quant_step_array[j-1]/2), allowed_act[j], x)
                    else:
                        x = torch.where((x >= (allowed_act[j] - quant_step_array[j-1]/2)) & (x < (allowed_act[j] + quant_step_array[j]/2)), allowed_act[j], x)
        return x

    def infer_traditional_act(self, x, num_bits, scale_values):

        count = 0

        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = quantize_traditional(x, num_bits, scale_values[count], signed=False)
                count += 1

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = quantize_traditional(x, num_bits, scale_values[count], signed=False)
                count += 1

        return x
