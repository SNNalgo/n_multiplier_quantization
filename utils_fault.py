import numpy as np
import torch

# S is a (2^n, n+1) matrix where the first n columns are the reverse binary representation of the row number
# and the last column is -1
def get_S_matrix(n):

    # initialize the matrix to be size 2^n x (n+1)
    S = np.zeros((2**n, n+1))

    # set last column to be -1
    S[:, -1] = -1

    # set the first n columns to be the reverse binary representation of the row number
    for i in range(2**n):
        S[i, 0:n] = np.array([int(x) for x in bin(i)[2:].zfill(n)[::-1]])

    return S

# S is a (2^n, n) matrix where the first n columns are the reverse binary representation of the row number
def get_S_matrix_activation(n):

    # initialize the matrix to be size 2^n x (n+1)
    S = np.zeros((2**n, n))

    # set the first n columns to be the reverse binary representation of the row number
    for i in range(2**n):
        S[i, 0:n] = np.array([int(x) for x in bin(i)[2:].zfill(n)[::-1]])

    return S

# R is a (n+1) x 1 vector going from 2^0, 2^1, 2^2, ..., 2^(n-1), zero_point
def get_R_vector(n, w_limit):

    zero_point = (2**n - 1)/2
    scale = w_limit/zero_point

    # R is a (n+1) x 1 vector going from 2^0, 2^1, 2^2, ..., 2^n
    R = np.zeros(n+1)
    for i in range(n):
        R[i] = 2**i
    R[-1] = zero_point

    # scale the R vector
    R = R * scale

    return R

# R is a n x 1 vector
def get_R_vector_activation(n, w_limit):

    zero_point = (2**n - 1)
    scale = w_limit/zero_point

    # R is a (n) x 1 vector going from 2^0, 2^1, 2^2, ..., 2^n
    R = np.zeros(n)
    for i in range(n):
        R[i] = 2**i

    # scale the R vector
    R = R * scale

    return R


def calc_dist(x, y):
    return torch.abs(x - y)

def calc_loss(dist, sigma):
    return 1 - 1*torch.exp(-(dist)**2/(sigma**2))

def get_closest_loss(p, levels, sigma):
    # takes in p which is a 3/4d tensor and levels which is a 1d tensor
    # returns loss and number of parameters
            
    for level in levels:
        if level == levels[0]:
            dist = calc_dist(p, level)
        else:
            dist = torch.min(dist, calc_dist(p, level))
        
    loss = calc_loss(dist, sigma)

    return torch.sum(loss), p.numel()

def get_closest_loss_mse(p, levels):
    # takes in p which is a 3/4d tensor and levels which is a 1d tensor
    # returns loss and number of parameters
            
    for level in levels:
        if level == levels[0]:
            dist = calc_dist(p, level)
        else:
            dist = torch.min(dist, calc_dist(p, level))

    # print("p", p)
    # print("dist", dist)
    loss = dist**2

    return torch.sum(loss), p.numel()

def get_closest_loss_w_fault(p, levels, sigma, validity):
    # takes in p which is a 3/4d tensor and levels which is a 1d tensor
    # returns loss and number of parameters

    cnt = 0
    for level in levels:
        #if level == levels[0]:
        if cnt == 0:
            dist = validity[cnt]*calc_dist(p, level) + (1-validity[cnt])*1e6
        else:
            dist = torch.min(dist, validity[cnt]*calc_dist(p, level) + (1-validity[cnt])*1e6)
        cnt = cnt+1
    loss = calc_loss(dist, sigma)

    return torch.sum(loss), p.numel()

def get_closest_loss_mse_w_fault(p, levels, validity):
    # takes in p which is a 3/4d tensor and levels which is a 1d tensor
    # returns loss and number of parameters

    cnt = 0
    for level in levels:
        #if level == levels[0]:
        if cnt == 0:
            dist = validity[cnt]*calc_dist(p, level) + (1-validity[cnt])*1e6
        else:
            dist = torch.min(dist, validity[cnt]*calc_dist(p, level) + (1-validity[cnt])*1e6)
        cnt = cnt+1

    # print("p", p)
    # print("dist", dist)
    loss = dist**2

    return torch.sum(loss), p.numel()

def quantize_traditional(v, num_bits, scale_factor, signed=True):

    if signed:
        q_n = 2**(num_bits-1)
        q_p = 2**(num_bits-1) - 1
    else:
        q_n = 0
        q_p = 2**num_bits - 1

    # scale v by scale_factor
    v = v / scale_factor

    v = torch.where(v <= -q_n, -q_n, v)
    v = torch.where(v >= q_p, q_p, v)

    v = torch.round(v)

    # scale v back
    v = v * scale_factor

    return v

def quantize_general(v, allowed_values, device):

    quant_step_array = torch.zeros(len(allowed_values)-1, device=device)

    for i in range(quant_step_array.size()[0]):
        quant_step_array[i] = allowed_values[i+1] - allowed_values[i]

    v = torch.clamp(v, allowed_values[0], allowed_values[-1])

    for j in range(allowed_values.size()[0]-1):
        if j == 0:
            v = torch.where(v < (allowed_values[j] + quant_step_array[j]/2), allowed_values[j], v)
        elif j == allowed_values.size()[0]-1:
            v = torch.where(v >= (allowed_values[j] - quant_step_array[j-1]/2), allowed_values[j], v)
        else:
            v = torch.where((v >= (allowed_values[j] - quant_step_array[j-1]/2)) & (v < (allowed_values[j] + quant_step_array[j]/2)), allowed_values[j], v)

    return v


def main():
    n = 4
    w_limit = 0.5

    print(get_S_matrix(n))
    print(get_R_vector(n, w_limit))

    print(get_S_matrix_activation(n))
    print(get_R_vector_activation(n, w_limit))

    # matrix multiplication of S and R
    print(np.dot(get_S_matrix(n), get_R_vector(n, w_limit)))
    print(np.dot(get_S_matrix_activation(n), get_R_vector_activation(n, w_limit)))

if __name__ == "__main__":
    main()

