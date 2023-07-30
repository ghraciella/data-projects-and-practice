import torch as nn 

'''
Draft classes for possible single tasks and multi-task models for me to implement

'''

class SemanticSegmentation(nn.Module):
    """linknet model """

    def __init__(self, filters=filters, classes=num_class, *args, **kwargs):
        super(SemanticSegmentation, self).__init__()
        self.encoder = SemanticSegmentationEncoder(filters)
        self.decoder = SemanticSegmentationDecoder( filters, classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DepthEstimation(nn.Module):
    """task of measuring the distance of each pixel relative to the camera """

    def __init__(self, filters=filters, classes=num_class, *args, **kwargs):
        super(DepthEstimation, self).__init__()
        self.encoder = DepthEstimationEncoder(filters)
        self.decoder = DepthEstimationDecoder( filters, classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SSDEMultinet(nn.Module):
    """ 
    SSDEMultinet: temporary name and draft structure for my multi-task network 
    with the first task as segmentation and second task as depth estimation 
    
    """
    def __init__(self, filters=filters, classes=num_class, *args, **kwargs):
        super(SSDEMultinet, self).__init__()

        self.shared_layer = SemanticSegmentationEncoder(filters),

        self.task1 = SemanticSegmentationDecoder( filters, classes)

        self.task2 = DepthEstimationDecoder( filters, classes)

    def forward(self, x):
        shared_encoder = self.shared_layer(x)
        out1 = self.task1(shared_encoder)
        out2 = self.task2(shared_encoder)
        return out1, out2














































































