import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as pltcm
import matplotlib.colors as mcolors
import wandb


from mtloc_misc_utils import *
from mtloc_mtltrain import *
#from mtloc_train import *
from mtloc_convert_util import *

cuda = torch.cuda.is_available()

def plot_img(data, label):
    fig, axs = plt.subplots(3, 3) # 9 images
    k = 0
    for i in range(3):
        for j in range(3):        
            axs[i, j].imshow(data[k].astype('uint8').reshape(28, 28))   # plot image            
            axs[i, j].set_ylabel("label:" + str(label[k].item()))       # print label
            k +=1
#plot_img(train_all_numpy, train_all_label_numpy)


def train_curves(epochs, train_losses, train_accuracies):
    iters = range(1, epochs+1)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    fig.suptitle('Training Curve')
    ax1.plot(iters, train_losses)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax2.plot(iters, train_accuracies, color = 'g')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Training Accuracy")
    plt.show()
 
   



def colorize(depth, cmap_name='magma', vmin=None, vmax=None, *args, **kwargs):

    """
    image: output data (from network) with size 1xHxW
    color map parameters: vmin, vmax, image
    default cmap color gradient: magma else ['viridis', 'plasma', 'inferno', 'magma', 'magma_r', 'cividis']
    bgr image -> rgb image 

    convert output(of floating no's  to rgb image data with integer values ranging from 0-255 via 
    applying out cmap, then convert image data to RGB from BGR
    """



    depth = depth.cpu().numpy()
 
    norm_depth = mcolors.Normalize(vmin = np.min(vmin), vmax = np.max(vmax), clip = False)
    norm_depth = norm_depth(depth)

    #print('norm shape', norm_depth.shape)
    apply_cmap = pltcm.get_cmap(cmap_name)
    bgr_image = apply_cmap(norm_depth, bytes=True)
    #print('bgr shape', bgr_image.shape)

    bgr255_image = bgr_image #*255
    bgrint8_image = bgr255_image.astype(np.uint8)
    rgb_image = cv2.cvtColor(bgrint8_image, cv2.COLOR_BGR2RGB)

    #rgb_image = bgrint8_image#bgr_image

    plt.imshow(rgb_image)
    plt.show()

    return rgb_image

def colorize_depth(image, cmap_name='magma', vmin=None, vmax=None):

    """
    image: output data (from network) with size 1xHxW
    color map parameters: vmin, vmax, image
    default cmap color gradient: magma else ['viridis', 'plasma', 'inferno', 'magma', 'magma_r', 'cividis']
    bgr image -> rgb image 

    convert output(of floating no's  to rgb image data with integer values ranging from 0-255 via 
    applying out cmap, then convert image data to RGB from BGR
    """

    normalized_image = mcolors.Normalize(vmin = np.min(vmin), vmax = np.max(vmax), clip = False)
    normalized_image = normalized_image(image)
    #normalized_image = np.ma.masked_invalid(normalized_image)

    apply_cmap = pltcm.get_cmap(cmap_name)
    bgr_image = apply_cmap(normalized_image, bytes=True)
    bgr255_image = bgr_image*255
    bgrint8_image = bgr255_image.astype(np.uint8)
    rgb_image = cv2.cvtColor(bgrint8_image, cv2.COLOR_BGR2RGB)
    #rgb_image = bgr_image

    return rgb_image



def viz_log(images, targets, predictions):

    num_plots = len(images)
    fig = plt.figure(figsize=(10,num_plots*5), dpi=240)
    out = []

    for idx in np.arange(num_plots): 
        ax = fig.add_subplot(3*num_plots, 3, 3*idx+1, xticks=[], yticks=[])

        npimg = images[idx]
        nptarget = targets[idx]
        nppred = predictions[idx]


        # outimage = colorize_depth(nppreds, cmap_name = 'viridis')
        # targetimage = colorize_depth(nptarget, cmap_name = 'viridis')

        plt.imshow(nppred)
        ax = fig.add_subplot(3*num_plots, 3, 3*idx+2, xticks=[], yticks=[])
        plt.imshow(nptarget)
        ax = fig.add_subplot(3*num_plots, 3, 3*idx+3, xticks=[], yticks=[])
        plt.imshow(npimg)

    plt.tight_layout(pad=0)
    return fig




def visualize(model, images, converter, targets):
    '''
    This function produces a matplotlib figure that shows the networks segmentation output 
    on images of the evaluation dataset
    '''
    num_plots = len(images)
    model.eval() # set network into test mode (turn off dropout)
    fig = plt.figure(figsize=(10,num_plots*5), dpi=240)
    out = []
    with torch.no_grad():
        if cuda==True:
            images = images.cuda()
        else:
            images = images
        out = model(images)
    for idx in np.arange(num_plots): 
        ax = fig.add_subplot(2*num_plots, 2, 2*idx+1, xticks=[], yticks=[])
        npimg = 255*images[idx].permute(1, 2, 0).cpu().numpy()
        nptarget = targets[idx].cpu().numpy()
        nppreds = torch.argmax(out[idx].float(), dim = 0).cpu().numpy()
        if idx > num_plots // 2 - 1:
            outimage = visualizeMask(nppreds, converter)
        else:
            outimage = visualizeMask(nppreds, converter, npimg.astype(np.uint8))
        targetimage = visualizeMask(nptarget, converter)
        plt.imshow(outimage)
        ax = fig.add_subplot(2*num_plots, 2, 2*idx+2, xticks=[], yticks=[])
        plt.imshow(targetimage)
    plt.tight_layout(pad=0)
    model.train() # back into training mode
    return fig






def visualize_depth_map(model, train_data, disp_labels):
    train_data, disp_labels = train_data, disp_labels

    #depth_colored = colorize_depth(depth_map_label, cmap_name = 'viridis')

    num_plots = len(train_data)
    model.eval() # set network into test mode (turn off dropout)
    fig = plt.figure(figsize=(10,num_plots*5), dpi=240)
    out = []
    with torch.no_grad():
        if cuda==True:
            train_data = train_data.cuda()
        else:
            train_data = train_data
        out = model(train_data)
    for idx in np.arange(num_plots): 
        ax = fig.add_subplot(3*num_plots, 3, 3*idx+1, xticks=[], yticks=[])

        npimg = train_data[idx].squeeze()
        nptarget = disp_labels[idx].squeeze()
        nppreds = out[idx].squeeze()

        #if idx > num_plots // 2 - 1:
        #    #outimage = visualizeMask(nppreds, converter)            

        #else:
        #    #outimage = visualizeMask(nppreds, converter, npimg.astype(np.uint8))


        outimage = colorize_depth(nppreds, cmap_name = 'viridis')


        #targetimage = visualizeMask(nptarget, converter)
        targetimage = colorize_depth(nptarget, cmap_name = 'viridis')

        plt.imshow(outimage)
        ax = fig.add_subplot(3*num_plots, 3, 3*idx+2, xticks=[], yticks=[])
        plt.imshow(targetimage)
        ax = fig.add_subplot(3*num_plots, 3, 3*idx+3, xticks=[], yticks=[])
        plt.imshow(npimg)

    plt.tight_layout(pad=0)
    model.train() # back into training mode
    return fig














if __name__ == "__main__": 



    # #to plot training curves

    # print("Training.............")

    # train_accuracies, train_losses = training(trainloader, epochs, model, loss_fn, optimizer)

    # print("begin visualization............check out your photos :)")

    # train_curves(epochs, train_losses, train_accuracies)

    # print("cool photos, okay bye!")


    #to test the colorize function

    root_dir_local ='C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes' #local machine
    root_dir = '/localdata/grace/Eze/Datasets/cityscapes/mtloc_cityscapes' #server

    # path1 = "C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes/leftImg8bit/train/bremen"
    # path2 = "C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes/disparity/train/bremen"

    path1 = "/localdata/grace/Eze/Datasets/cityscapes/mtloc_cityscapes/leftImg8bit/train/bremen"
    path2 = "/localdata/grace/Eze/Datasets/cityscapes/mtloc_cityscapes/disparity/train/bremen"

    testimage = "bremen_000004_000019"
    rgb_suffix = testimage + "_leftImg8bit.png"
    gt_suffix = testimage + "_disparity.png"
    testImageRGB = cv2.imread(os.path.abspath(os.path.join(path1 , rgb_suffix)))
    testGT = cv2.imread(os.path.abspath(os.path.join(path2 , gt_suffix)), 0)
    #cv2.namedWindow('rgbleftbit', cv2.WINDOW_NORMAL)
    #cv2.imshow("rgbleftbit", testImageRGB)
    #cv2.namedWindow('GT', cv2.WINDOW_NORMAL)
    #cv2.imshow("GT",testGT)
    outimage = colorize_depth(testGT, cmap_name='inferno', vmin=10, vmax=10)
    v_outimage = colorize_depth(testGT, cmap_name = 'viridis')
    c_outimage = colorize(testGT, cmap_name='magma', min_depth = 1e-3, max_depth = 80)

    cv2.namedWindow('outimage', cv2.WINDOW_NORMAL)
    cv2.imshow("outimage",outimage)

    cv2.namedWindow('v_outimage', cv2.WINDOW_NORMAL)
    cv2.imshow("v_outimage",v_outimage)

    cv2.namedWindow('c_outimage', cv2.WINDOW_NORMAL)
    cv2.imshow("c_outimage",c_outimage)

    cv2.waitKey(0)












