
import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchmetrics.functional import image_gradients
from torchmetrics.functional import structural_similarity_index_measure, multiscale_structural_similarity_index_measure



def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    
    # Point-wise depth
    l_depth = torch.mean(torch.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = image_gradients(y_true)
    dy_pred, dx_pred = image_gradients(y_pred)

    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = torch.clip((1 - structural_similarity_index_measure(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)
    


    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * torch.mean(l_edges)) + (w3 * torch.mean(l_depth))


class CombDepthLoss(nn.Module):
    """
    ToDo

    """

    def __init__(self):
        super(CombDepthLoss, self).__init__()

    def forward(self, y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):

        # Point-wise depth
        l_depth = torch.mean(torch.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = image_gradients(y_true)
        dy_pred, dx_pred = image_gradients(y_pred)

        l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = torch.clip((1 - structural_similarity_index_measure(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)
    


        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = theta

        loss = (w1 * l_ssim) + (w2 * torch.mean(l_edges)) + (w3 * torch.mean(l_depth))
        return loss








class MSELoss(nn.Module):
    """
    MSE Loss: l = (y_hat - y)^2

    """

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):

        diff = (output - target)**2
        loss = diff.sum()
        return loss







class DiceLoss(nn.Module):
    """
    We have the Dice_Coefficient = (2 TP)/(2 TP + FP + FN) = (2|X n Y|)/(|X| + |Y|)
    can then be defined as a loss function
    Dice_Loss(p,p') = 1 - (2p p' + 1)/(p + p' + 1) where p E [0,1] and 0<=p'<=1.

    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):

        smooth = 0.001

        iflat = output.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))



class HuberLoss(nn.Module):

    """
     For a batch of size, N,   l(x, y) = L = \{l_1, ..., l_N\}^T

    with

        l_n = 
        --> 0.5 (x_n - y_n)^2, if |x_n - y_n| < delta 
        --> delta * (|x_n - y_n| - 0.5 * delta),  otherwise 


    """

    def forward(self, predictions, targets, delta = 1.0, *args, **kwargs):
        loss = 0
        predictions , targets = predictions, targets
        for x, y in zip(predictions, targets):
            if torch.abs(x-y).any() < delta:
                loss += (0.5*(x-y)**2).mean()
            else:
                loss += (delta*torch.abs(x-y) - 0.5*delta).mean()

        loss =loss/predictions.shape[0]
        return loss

            
class BerhuLoss(nn.Module):

    """
     For a batch of size, N,   l(x, y) = L = \{l_1, ..., l_N\}^T

     x_n = y_hat
    with
        c = 1/5 x max_i(|x_n - y_n|)
        l_n = B(x_n) = 
        --> |x_n - y_n|, if |x_n - y_n| <= c 
        --> ((x_n - y_n)^2  + c^2)/2c,  otherwise 


    """
    
    def forward(self, predictions, targets,  *args, **kwargs):

        loss = 0
        for x, y in zip(predictions, targets):
            c = (torch.max(abs(x-y)).item())/5
            if abs(x-y) <= c:
                loss += (abs(x-y)).mean()
            else:
                loss += (((x-y)**2 + c**2)/(2*c)).mean()

        loss =loss/predictions.shape[0]
        return loss





if __name__ == "__main__":

     mse_loss = MSELoss()
     dice_loss = DiceLoss()
     huber_loss = HuberLoss()
     berhu_loss = BerhuLoss()

     output = torch.FloatTensor([0.82, 0.42, 0.79, 0.31])
     target = torch.FloatTensor([1., 0.61, 0.85, 0.50])

     M = mse_loss(output, target)
     print('MSE Loss: ', M)

     T = dice_loss(output, target)
     print('Dice Loss:', T)

     HL = huber_loss(output, target)
     print('Huber Loss:', HL)

     BL = berhu_loss(output, target)
     print('BerHu Loss:', BL)

     comb_loss = M + HL 
     print('MSE + Huberloss:', comb_loss)

     Hloss = nn.HuberLoss()
     Mloss = nn.MSELoss()

     H_loss = Hloss(output, target)
     M_loss = Mloss(output, target)
     print(f'torch Huber: {H_loss}, torch MSE: {M_loss}')

     #ssim, mssim
     preds = torch.rand([3, 3, 256, 256])     
     targs = preds * 0.75    
     ssim_loss = structural_similarity_index_measure(preds, targs)
     mssim_loss = multiscale_structural_similarity_index_measure(preds, targs, data_range=1.0)
     print(f'torch ssim loss: {ssim_loss}, mssim loss: {mssim_loss}')


     comb_depth_loss = depth_loss_function(targs, preds, theta=0.1, maxDepthVal=1000.0/10.0)
     print(f'torch combined depth loss: {comb_depth_loss}')


    







