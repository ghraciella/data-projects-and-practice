import torch
import torch.nn as nn
from torchvision.models import resnet
import torch.nn.functional as F



num_class = 19

filters = [16, 32, 64, 128, 256, 512] 





def Nonlinearity():
    actv = {'relu': nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(),
            'logsoftmax': nn.LogSoftmax(dim=1)
            }
    return actv



class ConvBatchnorm(nn.Module):
    """ To Do: class for both a conv-batchnorm sequence and a convtranspose-batchnorm sequence
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride , padding, output_padding=0, groups=1, bias=False , transposed = False, *args, **kwargs):
        super(ConvBatchnorm, self).__init__()
        if transposed==True:
            self.conv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding)
        else:
            self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride=stride, padding = padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.bn(self.conv(x))



class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, activation = 'relu', *args, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv_bn1 = ConvBatchnorm(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.nonlinearity = Nonlinearity()[activation]
        self.conv_bn2 = ConvBatchnorm(out_channels, out_channels, kernel_size, stride=1, padding=padding, groups=groups, bias=bias)
        self.downsample = None

        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv_bn1(x)
        out = self.nonlinearity(out)
        out = self.conv_bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlinearity(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BottleneckBlock, self).__init__()

        self.conv_bn1 = ConvBatchnorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=bias)
        self.nonlinearity = Nonlinearity()[activation]
        self.conv_bn2 = ConvBatchnorm(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.nonlinearity = Nonlinearity()[activation]
        self.conv_bn3 = ConvBatchnorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=bias)

        self.downsample = None

        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels))

        if stride != 1 or in_channels != self.expansion*out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv_bn1(x)
        out = self.nonlinearity(out)
        out = self.conv_bn2(out)
        out = self.nonlinearity(out)
        out = self.conv_bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlinearity(out)

        return out





class InitialBaseBlock(nn.Module):

    def __init__(self, in_channels=3, filters=filters, activation='relu', pretrained = False, bias=False, *args, **kwargs):
        super(InitialBaseBlock, self).__init__()
        #Todo, structure intial block as its own class

        self.filters = filters
        self.in_channels = in_channels

        if pretrained==True:
            #inital block, pretrained weights

            base = resnet.resnet18(pretrained=True)

            self.initial_block = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            )


        else:
            #inital block, no pretrained weights
            self.conv_bn1 = ConvBatchnorm(in_channels=self.in_channels, out_channels=self.filters[2], kernel_size=7, stride=2, padding=3, bias=bias)
            self.nonlinearity = Nonlinearity()[activation]
            self.pooling = nn.MaxPool2d(3, 2, 1)

            self.initial_block = nn.Sequential(self.conv_bn1,
                                               self.nonlinearity,
                                               self.pooling)

    def forward(self, x):

        #initial block

        x = self.initial_block(x)

        return x

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

    def forward(self, x):
        weights = torch.zeros(self.num_channels, 1, self.stride, self.stride)
        if torch.cuda.is_available():
            weights = weights.cuda()
        weights[:, :, 0, 0] = 1
        return F.conv_transpose2d(x, weights, stride=self.stride, groups=self.num_channels)



class DeConv(nn.Module):

    def __init__(self, in_channels,  kernel_size = 7, stride=2, activation='relu', *args, **kwargs):
        super(DeConv, self).__init__()

        self.nonlinearity = Nonlinearity()[activation]

        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1)//2
        self.output_padding = self.kernel_size % 2

        self.deconv_block1 = nn.Sequential(ConvBatchnorm(in_channels, in_channels//2, kernel_size=self.kernel_size, stride=stride, padding=self.padding, output_padding=self.output_padding, transposed=True),
                              self.nonlinearity)
        self.deconv_block2 = nn.Sequential(ConvBatchnorm(in_channels//2, in_channels//2, kernel_size=self.kernel_size, stride=stride, padding=self.padding, output_padding=self.output_padding, transposed=True),
                              self.nonlinearity)
        self.deconv_block3 =  nn.Sequential(ConvBatchnorm(in_channels//2, in_channels//2, kernel_size=self.kernel_size, stride=stride, padding=self.padding, output_padding=self.output_padding, transposed=True),
                              self.nonlinearity)


    def forward(self, x):
        x = self.deconv_block1(x)
        x = self.deconv_block2(x)
        x = self.deconv_block3(x)

        return x


class UpConv(nn.Module):

    def __init__(self, in_channels,  kernel_size = 5, stride=1, padding = 2, activation='relu', *args, **kwargs):
        super(UpConv, self).__init__()

        self.nonlinearity = Nonlinearity()[activation]
        self.unpool = Unpool(in_channels)
        self.upconv_block = nn.Sequential(self.unpool,
                              ConvBatchnorm(in_channels, in_channels//2, kernel_size=kernel_size, stride=stride, padding=padding),
                              self.nonlinearity)
        self.upconv1 = self.upconv_block(in_channels)
        self.upconv2 = self.upconv_block(in_channels//2)
        self.upconv3 = self.upconv_block(in_channels//4)
        self.upconv4 = self.upconv_block(in_channels//8)


    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x

class UpProjBlocks(nn.Module):

    def __init__(self, in_channels, activation='relu', *args, **kwargs):
        super(UpProjBlocks, self).__init__()
        
        self.nonlinearity = Nonlinearity()[activation]

        self.unpool = Unpool(in_channels)

        self.upper_branch = nn.Sequential(ConvBatchnorm(in_channels, in_channels//2, kernel_size=5, stride=1, padding=2),
                              self.nonlinearity,
                              ConvBatchnorm(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1))

        self.lower_branch = ConvBatchnorm(in_channels, in_channels//2, kernel_size=5, stride=1, padding=2)
                              

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.upper_branch(x)
        x2 = self.lower_branch(x)
        cat_x = x1 + x2
        out = self.nonlinearity(cat_x)
        return out

class FastUpConvBlocks(nn.Module):

    def __init__(self, in_channels,  activation='relu', *args, **kwargs):
        super(FastUpConvBlocks, self).__init__()

        self.nonlinearity = Nonlinearity()[activation]
        self.fastup_block1 = ConvBatchnorm(in_channels, in_channels//2, kernel_size=3),
        self.fastup_block2 = ConvBatchnorm(in_channels, in_channels//2, kernel_size=(2,3)),
        self.fastup_block3 = ConvBatchnorm(in_channels, in_channels//2, kernel_size=(3,2)),
        self.fastup_block4 = ConvBatchnorm(in_channels, in_channels//2, kernel_size=2),
        self.pixel_shuffle = nn.PixelShuffle(2)                      


    def forward(self, x):
        x1 = self.fastup_block1(nn.functional.pad(x, (1, 1, 1, 1)))
        x2 = self.fastup_block2(nn.functional.pad(x, (1, 1, 0, 1)))
        x3 = self.fastup_block3(nn.functional.pad(x, (0, 1, 1, 1)))
        x4 = self.fastup_block4(nn.functional.pad(x, (0, 1, 0, 1)))

        cat_x = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.pixel_shuffle(cat_x)
        out = self.nonlinearity(out)

        return out

class FastUpProjBlocks(nn.Module):

    def __init__(self, in_channels,  activation='relu', *args, **kwargs):
        super(FastUpProjBlocks, self).__init__()

        self.nonlinearity = Nonlinearity()[activation]

        self.upper_branch = nn.Sequential(FasterUpConvBlocks(in_channels),
                              self.nonlinearity,
                              ConvBatchnorm(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1))

        self.lower_branch = FasterUpConvBlocks(in_channels)
                              

    def forward(self, x):
        x1 = self.upper_branch(x)
        x2 = self.lower_branch(x)
        cat_x = x1 + x2
        out = self.nonlinearity(cat_x)
        return out


class UpProject(nn.Module):

    def __init__(self, in_channels, *args, **kwargs):
        super(UpProject, self).__init__()
        self.upproj_block = UpProjBlocks(in_channels)
        self.upproj1 = self.upproj_block(in_channels)
        self.upproj2 = self.upproj_block(in_channels//2)
        self.upproj3 = self.upproj_block(in_channels//4)
        self.upproj4 = self.upproj_block(in_channels//8)


    def forward(self, x):
        x = self.upproj1(x)
        x = self.upproj2(x)
        x = self.upproj3(x)
        x = self.upproj4(x)
        return x


class FasterUpConv(nn.Module):

    def __init__(self, in_channels,  *args, **kwargs):
        super(FasterUpConv, self).__init__()

        self.fasterupconv_block = FastUpConvBlocks(in_channels)
        self.fasterupconv1 = self.fasterupconv_block(in_channels)
        self.fasterupconv2 = self.fasterupconv_block(in_channels//2)
        self.fasterupconv3 = self.fasterupconv_block(in_channels//4)
        self.fasterupconv4 = self.fasterupconv_block(in_channels//8)


    def forward(self, x):
        x = self.fasterupconv1(x)
        x = self.fasterupconv2(x)
        x = self.fasterupconv3(x)
        x = self.fasterupconv4(x)
        return x


class FasterUpProject(nn.Module):

    def __init__(self, in_channels,  *args, **kwargs):
        super(FasterUpProject, self).__init__()

        self.fasterupproj_block = FastUpProjBlocks(in_channels)
        self.fasterupproj1 = self.fasterupproj_block(in_channels)
        self.fasterupproj2 = self.fasterupproj_block(in_channels//2)
        self.fasterupproj3 = self.fasterupproj_block(in_channels//4)
        self.fasterupproj4 = self.fasterupproj_block(in_channels//8)


    def forward(self, x):
        x = self.fasterupproj1(x)
        x = self.fasterupproj2(x)
        x = self.fasterupproj3(x)
        x = self.fasterupproj4(x)
        return x