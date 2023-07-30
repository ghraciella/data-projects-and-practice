import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import torchvision
from torchsummary import summary

from mtloc_load_dataset import *
from mtloc_model_helpers import *

classes = 10
#input_size = 784
#hidden_layers = [250,100]

num_class = 19

filters = [64, 128, 256, 512]




class simple_model(nn.Module):
    def __init__(self, input_features, output_features, n_classes = classes, activation = 'relu', *args, **kwargs):
        super().__init__()

        self.input_features, self.output_features = input_features, output_features
        self.n_classes = n_classes
        self.fc1 = nn.Linear(self.input_features, self.output_features)
        self.fc2 = nn.Linear(250, 100) #nn.Linear(output_features = 250, output_features= 100)  
        self.fc3 = nn.Linear(100, self.n_classes) 

        self.activation = activation
        self.activation_fn = Nonlinearity()[activation]
        self.probabilities = Nonlinearity()['logsoftmax']


    def forward(self, x):
        #print(x.shape)
        x = x.view(-1,self.input_features)
        x = self.fc1(x)
        x = self.activation_fn(x)
        #print(x.shape)
        x = self.fc2(x)
        x = self.activation_fn(x)
        x = self.fc3(x)
        x = self.probabilities(x)
        return x 





class ConvActivationBatchnorm(nn.Module):
    """ To Do: write documentation for this grace
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, transposed = False, stride = 1):
        super(ConvActivationBatchnorm, self).__init__()
        if transposed:
            outputpadding = 1
            self.conv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = outputpadding)
        else:
            self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False)
        self.activation = nn.ReLU()
        self.BN = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.BN(self.activation(self.conv(x)))



class FCN8Semantic(nn.Module):
    """FCN8 model """

    def __init__(self, filter=filter, classes=num_class, *args, **kwargs):
        super(FCN8Semantic, self).__init__()
        self.classes = classes
        self.filters =  filters
        self.conv_block1 = nn.Sequential(ConvActivationBatchnorm(in_channels=3, out_channels=self.filters[0], kernel_size=3, padding=1), 
                                         ConvActivationBatchnorm(in_channels=self.filters[0], out_channels=self.filters[0],kernel_size=3, padding=1))

        self.conv_block2 = nn.Sequential(ConvActivationBatchnorm(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=3, padding=1), 
                                         ConvActivationBatchnorm(in_channels=self.filters[1], out_channels=self.filters[1],kernel_size=3, padding=1))

        self.conv_block3 = nn.Sequential(ConvActivationBatchnorm(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=3, padding=1), 
                                         ConvActivationBatchnorm(in_channels=self.filters[2], out_channels=self.filters[2],kernel_size=3, padding=1),
                                         ConvActivationBatchnorm(in_channels=self.filters[2], out_channels=self.filters[2],kernel_size=3, padding=1))

        self.conv_block4 = nn.Sequential(ConvActivationBatchnorm(in_channels=self.filters[2], out_channels=self.filters[3], kernel_size=3, padding=1), 
                                         ConvActivationBatchnorm(in_channels=self.filters[3], out_channels=self.filters[3],kernel_size=3, padding=1),
                                         ConvActivationBatchnorm(in_channels=self.filters[3], out_channels=self.filters[3],kernel_size=3, padding=1))

        self.conv_block5 = nn.Sequential(ConvActivationBatchnorm(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=3, padding=1), 
                                         ConvActivationBatchnorm(in_channels=self.filters[3], out_channels=self.filters[3],kernel_size=3, padding=1),
                                         ConvActivationBatchnorm(in_channels=self.filters[3], out_channels=self.filters[3],kernel_size=3, padding=1))
        self.conv_dropout1 = nn.Sequential(ConvActivationBatchnorm(in_channels=self.filters[3], out_channels=self.filters[0], kernel_size=7, padding=3),
                                          nn.Dropout(0.1))
        self.conv_dropout2 = nn.Sequential(ConvActivationBatchnorm(in_channels=self.filters[0], out_channels=self.filters[2], kernel_size=1, padding=0),
                                          nn.Dropout(0.1))
        self.conv8 = ConvActivationBatchnorm(in_channels=self.filters[2], out_channels=self.filters[3], kernel_size=1, padding = 0)


        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.tranconv1 = ConvActivationBatchnorm(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=3, stride=2, padding=1, transposed=True)

        self.tranconv2 = ConvActivationBatchnorm(in_channels=2*self.filters[3], out_channels=self.filters[2], kernel_size=3, stride=2, padding=1, transposed=True)

        self.tranconv3 = nn.ConvTranspose2d(in_channels=2*self.filters[2], out_channels=classes, kernel_size=16, padding=4,  stride=8)

    def forward(self, x):
        x = self.downsample(self.conv_block1(x))
        x = self.downsample(self.conv_block2(x))
        x_pool3 = self.downsample(self.conv_block3(x)) 
        x_pool4 = self.downsample(self.conv_block4(x_pool3))
        x = self.downsample(self.conv_block5(x_pool4))
        x = self.conv_dropout1(x)
        x = self.conv_dropout2(x)
        x = self.conv8(x)
        x = self.tranconv1(x)
        x = torch.cat((x, x_pool4), dim = 1)
        x = self.tranconv2(x)
        x = torch.cat((x, x_pool3), dim = 1)
        x = self.tranconv3(x)
        return x









if __name__ == "__main__": 
    
    #model = CustomLinear(500, 250, bias=True)
    ##model.apply(initialize_weights)

    #sample_input = torch.randn(4)
    #print('custom linear', model(sample_input))

    #for parameter in model.named_parameters():
    #  print('parameter from custom linear', parameter)

    model =simple_model(784, 250, 10,  'lrelu')
    print('Forward propagation:', model)


    summary(model, (1, 28, 28))


    #for name, module in net.named_modules():
    #    print('modules for simple model', module)

    total_no_parameters = sum(p.numel() for p in model.parameters())
    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)


    print('total number of parameters: ', total_no_parameters)
    print('number of trainable parameters: ', total_trainable_parameters)

    num_class = 19
    filters = [64, 128, 256, 512]
    #print(FCN8Semantic(filters, num_class))
    #summary(FCN8Semantic(filters, num_class), (3, 1024, 512))


































