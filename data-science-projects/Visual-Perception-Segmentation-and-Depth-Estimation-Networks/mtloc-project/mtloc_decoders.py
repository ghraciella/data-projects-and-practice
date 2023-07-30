import torch
import torch.nn as nn
from mtloc_model_helpers import *


class LinkSegDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, activation='relu', *args, **kwargs):

        super(LinkSegDecoder, self).__init__()
        self.nonlinearity = Nonlinearity()[activation]

        self.convbn_nonlin1 = nn.Sequential(ConvBatchnorm(in_channels, in_channels//4, kernel_size=1, stride=1, padding=0, bias=bias),
                              self.nonlinearity)

        self.convbn_nonlin2 = nn.Sequential(ConvBatchnorm(in_channels//4, in_channels//4, kernel_size, stride, padding, output_padding=output_padding, bias=bias, transposed=True),
                              self.nonlinearity)

        self.convbn_nonlin3 = nn.Sequential(ConvBatchnorm(in_channels//4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
                              self.nonlinearity)

    def forward(self, x):
        x = self.convbn_nonlin1(x)
        x = self.convbn_nonlin2(x)
        x = self.convbn_nonlin3(x)

        return x





def decoder_type(in_channels, kernel_size = 7, stride=2, decoder_name = 'deconv', activation='relu', *args, **kwargs):

    if decoder_name == 'deconv':
        return DeConv(in_channels,  kernel_size=kernel_size, stride=stride, activation=activation)
    elif decoder_name == 'upconv':
        return UpConv(in_channels,  kernel_size = kernel_size, stride=stride, padding = 2, activation=activation)
    elif decoder_name == 'fasterupconv':
        return FasterUpConv(in_channels)
    elif decoder_name == 'upproj':
        return UpProject(in_channels)
    elif decoder_name == 'fasterupproj':
        return FasterUpProject(in_channels)
    else:
        print("{} is not a valid decoder for the network".format(decoder_name))
        assert False, "{} is not a valid decoder for the network".format(decoder_name)





class DepthDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, activation='relu', decoder_name='deconv', output_size =(512, 1024), *args, **kwargs):

        super(DepthDecoder, self).__init__()
        self.nonlinearity = Nonlinearity()[activation]
        self.output_size = output_size

        #decoder block
        self.decoder = decoder_type(self.filters[5]//2, kernel_size = kernel_size, stride=stride, decoder_name=decoder_name, activation=activation)
        
        self.conv = nn.Conv2d(self.filters[5]//4, 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.interpolation =  nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)



    def forward(self, x):
        x = self.decoder(x)

        x = self.conv(x)

        x = self.interpolation(x)

        return x





if __name__ == "__main__": 
    pass


