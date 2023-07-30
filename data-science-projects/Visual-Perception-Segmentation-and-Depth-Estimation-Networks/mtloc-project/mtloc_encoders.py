import torch
import torch.nn as nn
from mtloc_model_helpers import *



class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, *args, **kwargs):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_channels, out_channels, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class PretrainedEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, *args, **kwargs):
        super(PretrainedEncoder, self).__init__()
        self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_channels, out_channels, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x




if __name__ == "__main__":

    print('To Do')







