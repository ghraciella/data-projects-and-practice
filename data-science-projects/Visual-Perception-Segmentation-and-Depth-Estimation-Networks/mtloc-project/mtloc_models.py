
import torch
import torch as nn
from torchvision.models import resnet
from torchsummary import summary

from mtloc_encoders import *
from mtloc_decoders import *
from mtloc_convert_util import *



CLASSES = cityscapes_labelsWithTrainID
num_class = len(CLASSES) + 1
#num_class = 19

filters = [16, 32, 64, 128, 256, 512, 1024, 2048] 




class LinkNet(nn.Module):
    """
    LinkNet model architecture for semantic segmentation
    
    """

    def __init__(self, in_channels=3, num_class=num_class, filters=filters, activation='relu', pretrained = False, *args, **kwargs):

        super(LinkNet, self).__init__()

        self.nonlinearity = Nonlinearity()[activation]

        #Todo, structure intial block as its own class
        self.initial_baseblock = InitialBaseBlock(in_channels=in_channels, filters=filters, activation='relu', pretrained = pretrained, bias=False)
        self.filters = filters


        if pretrained:
            #Encoder block, pretrained weights

            base = resnet.resnet18(pretrained=True)
            self.encoder1 = base.layer1
            self.encoder2 = base.layer2
            self.encoder3 = base.layer3
            self.encoder4 = base.layer4

        else:
            #Encoder block, no pretrained weights

            self.encoder1 = Encoder(self.filters[2], self.filters[2], kernel_size=3, stride=1, padding=1)
            self.encoder2 = Encoder(self.filters[2], self.filters[3], kernel_size=3, stride=2, padding=1)
            self.encoder3 = Encoder(self.filters[3], self.filters[4], kernel_size=3, stride=2, padding=1)
            self.encoder4 = Encoder(self.filters[4], self.filters[5], kernel_size=3, stride=2, padding=1)


        #decoder
        self.decoder1 = LinkSegDecoder(self.filters[2], self.filters[2], kernel_size=3, stride=1, padding=1, output_padding=0)
        self.decoder2 = LinkSegDecoder(self.filters[3], self.filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = LinkSegDecoder(self.filters[4], self.filters[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = LinkSegDecoder(self.filters[5], self.filters[4], kernel_size=3, stride=2, padding=1, output_padding=1)

        # final block
        self.conv_nontp1 = nn.Sequential(ConvBatchnorm(self.filters[2], self.filters[1], kernel_size=3, stride=2, padding=1, output_padding=1, transposed=True),
                              self.nonlinearity)

        self.conv_non = nn.Sequential(ConvBatchnorm(self.filters[1], self.filters[1], kernel_size=3, stride=1, padding=1),
                              self.nonlinearity)

        self.conv_nontp2 = nn.Sequential(ConvBatchnorm(self.filters[1], num_class, kernel_size=2, stride=2, padding=0, transposed=True),
                              self.nonlinearity)

        self.log_probabilities = Nonlinearity()['logsoftmax']

    def forward(self, x):
        # Initial base block

        x = self.initial_baseblock(x)

        # Encoder blocks
        enc_1 = self.encoder1(x)
        enc_2 = self.encoder2(enc_1)
        enc_3 = self.encoder3(enc_2)
        enc_4 = self.encoder4(enc_3)

        # Decoder blocks
        dec_4 = enc_3 + self.decoder4(enc_4)
        dec_3 = enc_2 + self.decoder3(dec_4)
        dec_2 = enc_1 + self.decoder2(dec_3)
        dec_1 = x + self.decoder1(dec_2)

        # final block 
        out = self.conv_nontp1(dec_1)
        out = self.conv_non(out)
        out = self.conv_nontp2(out)

        out = self.log_probabilities(out)

        return out






class DisparityNet(nn.Module):
    def __init__(self, in_channels = 3, kernel_size = 7, stride=2, num_class=num_class, filters=filters, activation='relu', decoder_name = 'deconv', pretrained=False, output_size =(512, 1024), *args, **kwargs):

        super(DisparityNet, self).__init__()

        self.nonlinearity = Nonlinearity()[activation]
        self.initial_baseblock = InitialBaseBlock(in_channels=in_channels, filters=filters, activation=activation, pretrained = pretrained, bias=False)

        self.filters = filters
        self.output_size = output_size

        if pretrained:
            #Encoder block, pretrained weights

            base = resnet.resnet18(pretrained=True)
            self.encoder1 = base.layer1
            self.encoder2 = base.layer2
            self.encoder3 = base.layer3
            self.encoder4 = base.layer4

        else:
            #Encoder block, no pretrained weights

            self.encoder1 = Encoder(self.filters[2], self.filters[2], kernel_size=3, stride=1, padding=1)
            self.encoder2 = Encoder(self.filters[2], self.filters[3], kernel_size=3, stride=2, padding=1)
            self.encoder3 = Encoder(self.filters[3], self.filters[4], kernel_size=3, stride=2, padding=1)
            self.encoder4 = Encoder(self.filters[4], self.filters[5], kernel_size=3, stride=2, padding=1)
            

        self.conv_bn  = ConvBatchnorm(self.filters[5], self.filters[5]//2, kernel_size=1, stride=1, padding=0)

        #decoder block
        self.decoder = decoder_type(self.filters[5]//2, kernel_size = kernel_size, stride=stride, decoder_name=decoder_name, activation=activation)
        
        self.conv = nn.Conv2d(self.filters[5]//4, 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.interpolation =  nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)


    def forward(self, x):
        
        # Initial base block

        x = self.initial_baseblock(x)
        #print(x.shape)

        # Encoder blocks
        enc_1 = self.encoder1(x)
        enc_2 = self.encoder2(enc_1)
        enc_3 = self.encoder3(enc_2)
        enc_4 = self.encoder4(enc_3)

        #print('1,2,3,4: ', enc_1.shape, enc_2.shape, enc_3.shape, enc_4.shape)
        #intermediate layer
        out = self.conv_bn(enc_4)
        #print(out.shape)

        # Decoder blocks
        out = self.decoder(out)
        #print(out.shape)

        out = self.conv(out)
        #print(out.shape)

        out = self.interpolation(out)
        #print(out.shape)
      
        return out


if __name__ == "__main__":

    num_class = 19

    filters = [16, 32, 64, 128, 256, 512, 1024, 2048] 

    #Linknet semantic segmentation
    pretrained_linkmodel = LinkNet(num_class=num_class, filters=filters, activation='relu', pretrained = True)
    linkmodel = LinkNet(num_class=num_class, filters=filters, activation='relu', pretrained = False)
    #print(linkmodel)
    summary(linkmodel, (3, 1024, 512))

    #Depth Estimation Net
    depthest_net = DisparityNet(in_channels = 3, num_class=num_class, filters=filters, activation='relu', decoder_name = 'deconv', pretrained=True, output_size =(512, 1024))
    #print(depthest_net)
    #summary(depthest_net, (3, 1024, 512))











