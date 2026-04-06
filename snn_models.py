import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, utils 

class MLP_Head(nn.Module):
    def __init__(self, in_size=512, out_size=10, beta=0.5):

        super().__init__()

        self.out_size = out_size
        self.in_size = in_size
        self.beta = beta

        spike_grad = surrogate.atan()
        
        self.net = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.in_size, self.in_size),
                    nn.BatchNorm1d(self.in_size),
                    snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Linear(self.in_size, self.output_neurons),
                    snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True)
                    )
    
    def forward(self, data):

        spk_rec = []

        for step in range(data.size(1)):
            spk_out = self.net(data[:,step,:])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

class LC(nn.Module):

    def __init__(self, in_size=512, out_size=10, beta=0.5):

        super().__init__()

        self.out_size = out_size
        self.in_size = in_size
        self.beta = beta

        spike_grad = surrogate.atan()
        
        self.net = nn.Sequential(
                    nn.Linear(self.in_size, self.out_size),
                    snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True)
                    )
    
    def forward(self, data):

        spk_rec = []
        utils.reset(self.net)

        for step in range(data.size(1)):
            spk_out = self.net(data[:,step,:])
            spk_rec.append(spk_out)
        return torch.stack(spk_rec)

class SNN_VGG(nn.Module):

    def __init__(self, beta=0.5, num_classes=10):
        super().__init__()

        spike_grad = surrogate.atan()
        beta = beta
        
        # conv, BN, leaky      
        # "D": [64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M", 512, 512, "M", 512 512, "M", 512, 512, "M"],
        
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.LayerNorm((64, 128, 128), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LayerNorm((64, 64, 64), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LayerNorm((128, 64, 64), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LayerNorm((128, 32, 32), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LayerNorm((256, 32, 32), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LayerNorm((256, 16, 16), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LayerNorm((256, 16, 16), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(256, 256, 3, padding=1),          
            nn.MaxPool2d(2),
            nn.LayerNorm((256, 8, 8), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LayerNorm((512, 8, 8), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LayerNorm((512, 4, 4), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LayerNorm((512, 4, 4), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LayerNorm((512, 2, 2), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LayerNorm((512, 2, 2), elementwise_affine=False),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LayerNorm((512, 1, 1), elementwise_affine=False),
            nn.Flatten(),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(512, num_classes),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
            )

    def forward(self, data): 
        spk_rec = []

        # data is of the form [batch_size, time_steps, C, H, W]

        utils.reset(self.net)


        for step in range(data.size(1)):

            resized_data = F.interpolate(data[:,step,:,:,:], size=(128, 128), mode='bilinear')
            spk_out = self.net(resized_data)
            spk_rec.append(spk_out)
        
        # returns [time_steps, batch_size, out_size]

        return torch.stack(spk_rec)
