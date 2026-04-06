"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import torch
import torch.nn.functional as F
import math
from quan_base_nmult import _Conv2dQ, Qmodes, _LinearQ, _ActQ


__all__ = ['Conv2dNMult', 'LinearNMult', 'ActLSQ']

class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
            -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


class QuantNmult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, S):
        allowed_weight = torch.matmul(alpha, S.T) - alpha[-1]
        w_q = torch.clamp(weight, allowed_weight[0], allowed_weight[-1])
        for j in range(allowed_weight.size()[0]):
            if j == 0:
                w_q = torch.where(w_q < (allowed_weight[j] + (allowed_weight[j+1]-allowed_weight[j])/2), allowed_weight[j], w_q)
            elif j == allowed_weight.size()[0]-1:
                step = (allowed_weight[j]-allowed_weight[j-1])
                w_q = torch.where(w_q >= (allowed_weight[j] - step/2), allowed_weight[j], w_q)
            else:
                step_h = (allowed_weight[j+1]-allowed_weight[j])
                step_l = (allowed_weight[j]-allowed_weight[j-1])
                w_q = torch.where((w_q >= (allowed_weight[j] - step_l/2)) & (w_q < (allowed_weight[j] + step_h/2)), allowed_weight[j], w_q)
        return w_q
    
    @staticmethod
    def backward(ctx, grad_weight):
        return grad_weight, None, None

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dNMult(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, **kwargs):
        super(Conv2dNMult, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            #base_vals = 2 ** torch.arange(self.nbits)
            scale = 2 * self.weight.abs().mean() / math.sqrt(Qp)
            self.alpha.data.copy_(scale*self.base_vals)
            self.init_state.fill_(1)
        #g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        #alpha = grad_scale(self.alpha, g)
        #w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        w_q = QuantNmult.apply(self.weight, self.alpha, self.S)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearNMult(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearNMult, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            #base_vals = 2 ** torch.arange(self.nbits)
            scale = 2 * self.weight.abs().mean() / math.sqrt(Qp)
            self.alpha.data.copy_(scale*self.base_vals)
            self.init_state.fill_(1)
        #g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        #alpha = grad_scale(self.alpha, g)
        #w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        w_q = QuantNmult.apply(self.weight, self.alpha, self.S)
        return F.linear(x, w_q, self.bias)


class ActLSQ(_ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(ActLSQ, self).__init__(nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        # x = x / alpha
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x
