import os
import cv2
import torch
import torch.nn as nn
import math
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
from torchsummary import summary







# Initializing parameters (weights and/or biases)

def init_params(self, init_type = 'xavier_uniform'):
    """
    initialize the parameters(:weight and bias) of the layers
    """

    for m in self.modules():

        if isinstance(m, nn.Conv2d) and init_type == 'xavier_norm':
            nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Conv2d) and init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Conv2d) and init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) 

        if init_type == 'default':
            pass

        else:
             assert 0, "Unsupported initialization type: {}".format(init_type)



 #Scheduler

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Adjust learning rate: Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        #print('LR is set to {}'.format(lr))
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer






# Transforms that can be done on the Data


data_size = (1024, 1024)

class data_augmentations(object):
    def __init__(self):
        # self.pilimage = transforms.ToPILImage()
        self.randomcrop = transforms.RandomCrop(data_size)
        self.randomhflip = transforms.RandomHorizontalFlip(p=0.5)
        # self.randomvflip = transforms.RandomVerticalFlip(p=0.5)
        # self.colorjitter = transformsColor_Jitter(brightness = 2)
        # self.ttensor = transforms.ToTensor()


    def __call__(self, file):
        # file = self.pilimage(file)
        file = self.randomcrop(file)
        file = self.randomhflip(file)
        # file = self.randomvflip(file)
        # file = self.colorjitter(file)
        # file = self.ttensor(file)

        return file







def data_camera_params(dataloader, *args, **kwargs):

        #TODO extracting sample pairs for batches during training

        image, sem_label, disp_label, baseline, focal_length, xp, yp, zp, up, vp, fy = next(iter(dataloader))

        #TODO get image width and height
        return image, sem_label, disp_label, baseline, focal_length, xp, yp, zp, up, vp, fy

def intrinsic_matrixs_coords(focal_length, xp, yp, zp, up, vp, fy, *args, **kwargs):

        fx = focal_length 

        #known intrinsic camera matrix

        #Todo maybe compute the inverse of the square matrix if it is invertible via torch.linalg.inv_ex(A) or torch.linalg.inv(M)


        K = np.array([
            [fx, 0, up, 0],
            [0, fy, vp, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype = np.float32)

        #transposed 3d co-ords

        XP = np.array([up, vp, 1], dtype = np.float32)
        #XP = torch.transpose(XP, 0, 1)
        XP = XP.T
        return K, XP


def zip_and_add(list1, list2):
    return [x+y for x,y in zip(list1,list2)]

def zip_and_subtract(list1, list2):
    return [x-y for x,y in zip(list1,list2)]

def zip_and_multiply(list1, list2):
    return [a*b for a, b in zip(list1, list2)]

def zip_and_divide(list1, list2):
    return [x/y for x,y in zip(list1,list2)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Disp2Depth(torch.nn.Module):

    '''
    convert disparity images to depth ground truth map (disparity is inversely proportional to depth):  
        
    depth = baseline * focal / disparity
    '''


    def __init__(self, baseline, focal_length, depth_scale=0.256, batch_size=1, scale=True,  *args, **kwargs):
        super().__init__()
        self.baseline = baseline
        self.focal_length = focal_length
        self.depth_scale = depth_scale
        self.scale = scale
        self.batch_size = batch_size

    def forward(self, disparity):

        if self.scale == True:
            disparity = disparity/self.depth_scale
        else:
            disparity = disparity

        disparity = disparity.detach().cpu().numpy()

        if np.isinf(disparity).any() == True:
            mask = np.isinf(disparity)
            disparity[mask] = 1

        if np.isnan(disparity).any() == True:
            mask = np.isnan(disparity)
            disparity[mask] = 1


        if self.batch_size > 1:
            base_focal_mul = zip_and_multiply(self.baseline,  self.focal_length)
            base_x_focal = [bf.cpu().numpy() for bf in base_focal_mul]
            base_x_focal = torch.tensor(np.array(base_x_focal, dtype=np.float32), device=device)
            e_disparity = (disparity + 1)
            e_disparity = torch.tensor(e_disparity, device=device)

            depth_div  = zip_and_divide(base_x_focal, e_disparity)
            depth_map = [d.cpu().numpy() for d in depth_div]
            depth_map  = torch.tensor(depth_map, device=device).requires_grad_()

        else:
            depth_map = (self.baseline * self.focal_length)/((disparity) + 1)
            depth_map = torch.tensor(depth_map, device=device) 

        return depth_map

class DepthNorm(torch.nn.Module):
    def __init__(self, min_depth = 1e-3, max_depth = 80, batch_size=1, *args, **kwargs):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.batch_size = batch_size

    #def __call__(self, depth):
    def forward(self, depth):

        if self.batch_size >= 1:
            depmin = [pix_i - self.min_depth for pix_i in depth.to(device)]  

            minmax = self.max_depth - self.min_depth  
            normalised_depth  =  [pix_j - minmax for pix_j in depmin] 
            normalised_depth = [nd.cpu().numpy() for nd in normalised_depth]
            normalised_depth = torch.tensor(normalised_depth, device=device).requires_grad_()

        else:
            normalised_depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
            normalised_depth = torch.tensor(normalised_depth, device=device) #.type_as(depth)

        return normalised_depth




class Disp2DepthNorm(torch.nn.Module):

    '''
    convert disparity images to depth ground truth map (disparity is inversely proportional to depth):  
        
    depth = baseline * focal / disparity
    '''


    def __init__(self, baseline, focal_length, depth_scale=0.256, min_depth = 1e-3, max_depth = 80, batch_size=1, scale=True, norm=True, *args, **kwargs):
        super().__init__()
        self.baseline = baseline
        self.focal_length = focal_length
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale = scale
        self.batch_size = batch_size
        self.norm = norm

    def forward(self, disparity):

        if self.scale == True:
            disparity = disparity/self.depth_scale
        else:
            disparity = disparity

        disparity = disparity.detach().cpu().numpy()
        #disparity = disparity.to(device)

        if np.isinf(disparity).any() == True:
            mask = np.isinf(disparity)
            disparity[mask] = 1

        if np.isnan(disparity).any() == True:
            mask = np.isnan(disparity)
            disparity[mask] = 1


        if self.batch_size > 1 and self.norm == True:
            base_focal_mul = zip_and_multiply(self.baseline,  self.focal_length)
            base_x_focal = [bf.cpu().numpy() for bf in base_focal_mul]
            base_x_focal = torch.tensor(np.array(base_x_focal, dtype=np.float32), device=device)
            e_disparity = (disparity + 1)
            e_disparity = torch.tensor(e_disparity, device=device)


            depth_div  = zip_and_divide(base_x_focal, e_disparity)
            depth_map = [d.cpu().numpy() for d in depth_div]
            depth_map  = torch.tensor(depth_map, device=device)#.requires_grad_()


            depmin = [pix_i - self.min_depth for pix_i in depth_map.to(device)]  

            minmax = self.max_depth - self.min_depth  
            normalised_depth  =  [pix_j - minmax for pix_j in depmin] 
            normalised_depth = [nd.cpu().numpy() for nd in normalised_depth]
            normalised_depth = torch.tensor(normalised_depth, device=device).requires_grad_()
            depth = normalised_depth
            return depth

        elif self.batch_size == 1 and self.norm == True:
            depth_map = (self.baseline * self.focal_length)/((disparity) + 1)
            depth_map = torch.tensor(depth_map, device=device) 

            normalised_depth = (depth_map - self.min_depth) / (self.max_depth - self.min_depth)
            normalised_depth = torch.tensor(normalised_depth, device=device) 
            depth = normalised_depth
            return depth

        else:
            depth_map = (self.baseline * self.focal_length)/((disparity) + 1)
            depth_map = torch.tensor(depth_map, device=device) 

            depth = depth_map
            return depth











if __name__ == "__main__":

    print('TO DO: Add other supporting functions')



    testimage_baseline = 0.209313
    testimage_focal_length = 2262.52
    
    #load test image and display
    path1 = "C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes/leftImg8bit/train/bremen"
    path2 = "C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes/disparity/train/bremen"
    testimage = "bremen_000004_000019"
    rgb_suffix = testimage + "_leftImg8bit.png"
    gt_suffix = testimage + "_disparity.png"
    testImageRGB = cv2.imread(os.path.abspath(os.path.join(path1 , rgb_suffix)))
    testGT = cv2.imread(os.path.abspath(os.path.join(path2 , gt_suffix)), 0)
    #cv2.namedWindow('rgbleftbit', cv2.WINDOW_NORMAL)
    #cv2.imshow("rgbleftbit", testImageRGB)

    cv2.namedWindow('GT', cv2.WINDOW_NORMAL)
    cv2.imshow("GT",testGT)

    #convert disparity to depth
    depth_map = disparity_to_depth(testGT, testimage_baseline, testimage_focal_length, depth_scale=256.)
    cv2.namedWindow('depth map', cv2.WINDOW_NORMAL)
    cv2.imshow("depth map", depth_map)


    ##colorize disparity and depth then display
    #disp_outimage = colorize_depth(testGT, cmap_name = 'viridis')
    #depth_outimage = colorize_depth(depth_map, cmap_name = 'viridis')

    #cv2.namedWindow('disp_outimage', cv2.WINDOW_NORMAL)
    #cv2.imshow("disp_outimage",disp_outimage)

    #cv2.namedWindow('depth_outimage', cv2.WINDOW_NORMAL)
    #cv2.imshow("depth_outimage",depth_outimage)
    cv2.waitKey(0)



