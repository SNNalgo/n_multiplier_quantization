import torch
import torch.nn as nn
import torch.nn.functional as F


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from utils import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def forward_act(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        inter_act = out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, inter_act
    
    def forward_clamp(self, x, allowed_act):

        identity = x

        max_act = torch.max(allowed_act)
        min_act = torch.min(allowed_act)
        max_tolerance_a = (allowed_act[-1] - allowed_act[-2])/2

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = torch.clamp(out, min_act, max_act + max_tolerance_a)

        inter_act = out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, inter_act
    
    def infer(self, x, allowed_act):

        quant_step_array = torch.zeros(len(allowed_act)-1)

        # calculate quant step array from allowed weight
        for i in range(len(allowed_act)-1):
            quant_step_array[i] = allowed_act[i+1] - allowed_act[i]

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = torch.clamp(out, allowed_act[0], allowed_act[-1])

        for j in range(len(allowed_act)):
            if j == 0:
                out = torch.where(out< (allowed_act[j] + quant_step_array[j]/2), allowed_act[j], out)
            elif j == len(allowed_act)-1:
                out = torch.where(out >= (allowed_act[j] - quant_step_array[j-1]/2), allowed_act[j], out)
            else:
                out = torch.where((out >= (allowed_act[j] - quant_step_array[j-1]/2)) & (out < (allowed_act[j] + quant_step_array[j]/2)), allowed_act[j], out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def infer_traditional_act(self, x, num_bits, scale_factor_1, scale_factor_2, quantize_last=True):
        
        identity = x
        #print(scale_factor_1, scale_factor_2)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = quantize_traditional(out, num_bits, scale_factor_1, signed=False)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if quantize_last:
            out = quantize_traditional(out, num_bits, scale_factor_2, signed=False)
        else:
            out = out

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if last_activation == "relu":
            self.last_activation = nn.ReLU(inplace=True)
        elif last_activation == "none":
            self.last_activation = nn.Identity()
        elif last_activation == "sigmoid":
            self.last_activation = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        return out
    
    def forward_act(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        inter_act = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        return out, inter_act
    
    def forward_clamp(self, x, allowed_act):

        identity = x

        max_act = torch.max(allowed_act)
        min_act = torch.min(allowed_act)
        max_tolerance_a = (allowed_act[-1] - allowed_act[-2])/2

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = torch.clamp(out, min_act, max_act + max_tolerance_a)
        inter_act = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        return out, inter_act
    
    def infer(self, x, allowed_act):

        quant_step_array = torch.zeros(len(allowed_act)-1)

        # calculate quant step array from allowed weight
        for i in range(len(allowed_act)-1):
            quant_step_array[i] = allowed_act[i+1] - allowed_act[i]

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = torch.clamp(out, allowed_act[0], allowed_act[-1])

        for j in range(len(allowed_act)):
            if j == 0:
                out = torch.where(out< (allowed_act[j] + quant_step_array[j]/2), allowed_act[j], out)
            elif j == len(allowed_act)-1:
                out = torch.where(out >= (allowed_act[j] - quant_step_array[j-1]/2), allowed_act[j], out)
            else:
                out = torch.where((out >= (allowed_act[j] - quant_step_array[j-1]/2)) & (out < (allowed_act[j] + quant_step_array[j]/2)), allowed_act[j], out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        return out
    
    def infer_traditional_act(self, x, num_bits, scale_factor_1, scale_factor_2, scale_factor_3, quantize_last=True):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = quantize_traditional(out, num_bits, scale_factor_1, signed=False)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = quantize_traditional(out, num_bits, scale_factor_2, signed=False)
         
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        if quantize_last:
            out = quantize_traditional(out, num_bits, scale_factor_3, signed=False)
        else:
            out = out

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes: int = 1000,
        num_channels=3,
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_activation="relu",
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # self._last_activation = last_activation

        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            num_channels,
            num_out_filters,
            kernel_size=7,
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block,
            num_out_filters,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block,
            num_out_filters,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block,
            num_out_filters,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            last_activation=last_activation,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_out_filters * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, last_activation="relu"
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                last_activation=(last_activation if blocks == 1 else "relu"),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    last_activation=(last_activation if i == blocks - 1 else "relu"),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)    
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x

    def forward_act(self, x):

        activations = []

        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        activations.append(x)

        x = self.maxpool(x)


        for layer in self.layer1:
            x, inter_act = layer.forward_act(x)
            activations.append(inter_act)
            activations.append(x)

        for layer in self.layer2:
            x, inter_act = layer.forward_act(x)
            activations.append(inter_act)
            activations.append(x)

        for layer in self.layer3:
            x, inter_act = layer.forward_act(x)
            activations.append(inter_act)
            activations.append(x)

        for layer in self.layer4:
            x, inter_act = layer.forward_act(x)
            activations.append(inter_act)
            activations.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x, activations

    def forward_clamp(self, x, allowed_act):

        activations = []

        max_act = torch.max(allowed_act)
        min_act = torch.min(allowed_act)
        max_tolerance_a = (allowed_act[-1] - allowed_act[-2])/2

        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = torch.clamp(x, min_act, max_act + max_tolerance_a)
        activations.append(x)

        x = self.maxpool(x)

        for layer in self.layer1:
            x, inter_act = layer.forward_clamp(x, allowed_act)
            activations.append(inter_act)
            activations.append(x)

        for layer in self.layer2:
            x, inter_act = layer.forward_clamp(x, allowed_act)
            activations.append(inter_act)
            activations.append(x)

        for layer in self.layer3:
            x, inter_act = layer.forward_clamp(x, allowed_act)
            activations.append(inter_act)
            activations.append(x)

        for layer in self.layer4:
            x, inter_act = layer.forward_clamp(x, allowed_act)
            activations.append(inter_act)
            activations.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x, activations

    def infer(self, x, allowed_act):

        quant_step_array = torch.zeros(len(allowed_act)-1)

        # calculate quant step array from allowed weight
        for i in range(len(allowed_act)-1):
            quant_step_array[i] = allowed_act[i+1] - allowed_act[i]

        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = torch.clamp(x, allowed_act[0], allowed_act[-1])

        for j in range(len(allowed_act)):
            if j == 0:
                x = torch.where(x< (allowed_act[j] + quant_step_array[j]/2), allowed_act[j], x)
            elif j == len(allowed_act)-1:
                x = torch.where(x >= (allowed_act[j] - quant_step_array[j-1]/2), allowed_act[j], x)
            else:
                x = torch.where((x >= (allowed_act[j] - quant_step_array[j-1]/2)) & (x < (allowed_act[j] + quant_step_array[j]/2)), allowed_act[j], x)
        
        x = self.maxpool(x)

        for layer in self.layer1:
            x = layer.infer(x, allowed_act)

        for layer in self.layer2:
            x = layer.infer(x, allowed_act)

        for layer in self.layer3:
            x = layer.infer(x, allowed_act)

        for layer in self.layer4:
            x = layer.infer(x, allowed_act)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
        
    def infer_traditional_act(self, x, num_bits, scale_factors_activations, quantize_fl):

        count = 0

        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if not quantize_fl:
            x = x
        else:
            x = quantize_traditional(x, num_bits, scale_factors_activations[count], signed=False)

        count += 1

        x = self.maxpool(x)

        for layer in self.layer1:
            x = layer.infer_traditional_act(x, num_bits, scale_factors_activations[count], scale_factors_activations[count+1])
            count += 2

        for layer in self.layer2:
            x = layer.infer_traditional_act(x, num_bits, scale_factors_activations[count], scale_factors_activations[count+1])
            count += 2

        for layer in self.layer3:
            x = layer.infer_traditional_act(x, num_bits, scale_factors_activations[count], scale_factors_activations[count+1])
            count += 2

        for layer in self.layer4:

            if layer == self.layer4[-1] and not quantize_fl:
                x = layer.infer_traditional_act(x, num_bits, scale_factors_activations[count], scale_factors_activations[count+1], quantize_last=False)
                count += 2
            else:
                x = layer.infer_traditional_act(x, num_bits, scale_factors_activations[count], scale_factors_activations[count+1])
                count += 2

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs), 512


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs), 2048


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs), 2048


def resnet50x2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs), 4096


def resnet50x4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs), 8192


def resnet50x5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs), 10240


def resnet200x2(**kwargs):
    return ResNet(Bottleneck, [3, 24, 36, 3], widen=2, **kwargs), 4096
