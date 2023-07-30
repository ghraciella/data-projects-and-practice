import sys
sys.path.append("..")

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.backends import cudnn


from datetime import datetime

from mtloc_cityscapes_dataset import *
from mtloc_convert_util import *
from mtloc_models import *
from mtloc_misc_utils import *
from mtloc_metrics import *
from mtloc_optimizers import *
from mtloc_objective_losses import *
from mtloc_visualise import *
cuda = torch.cuda.is_available()

#logging training
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
#log_dir = 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_runs/train' + current_time
log_dir = '/localdata/grace/Eze/Project/mtl_runs/train' + current_time  #remote server 

cuda = torch.cuda.is_available()
EPOCHS = 10
LOG_FREQUENCY = 100
lr = 0.001
weight_decay = 1e-4
momentum = 0.9
filters = [16, 32, 64, 128, 256, 512] 

CLASSES = cityscapes_labelsWithTrainID
num_of_classes = len(CLASSES) + 1


#model_name = 'linknet'

#transformations on images and target masks
# transform = transforms.ToTensor()
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


target_transform = ClassConverter(CLASSES)
#target_transform = ClassConverter([7, 9, 26, 27, 28])

#data augumentation
augmentations = data_augmentations()


# this is a dict to return the original class ids from our training labels
inverse_class_converter = target_transform.get_inverter()



transcityscape = CityscapesDepthSeg('/localdata/grace/Eze/Datasets/cityscapes/mtloc_cityscapes',
                                split='train',
                                mode='fine',
                                target_type='semantic',
                                augumentation=augmentations,
                                transform=None,
                                image_transform=transform,
                                size=data_size,
                                target_transform=target_transform)

train_dataloader = DataLoader(transcityscape, batch_size=1,
                                 shuffle=True, num_workers=4, collate_fn=None)

val_cityscape = CityscapesDepthSeg('/localdata/grace/Eze/Datasets/cityscapes/mtloc_cityscapes',
                                split='val',
                                mode='fine',
                                target_type='semantic',
                                transform=None,
                                image_transform=transform,
                                size=data_size,
                                target_transform=target_transform)

val_dataloader = DataLoader(val_cityscape, batch_size=1,
                                 shuffle=True, num_workers=4, collate_fn=None)




#initializing model network and use of GPU if found

# model = FCN8Semantic(len(CLASSES) + 1)
# model = LinkNet(len(CLASSES) + 1)

model = LinkNet(num_class=num_of_classes, filters=filters, activation='relu', pretrained = False)

#model = DisparityNet(in_channels = 3, num_class=num_class, filters=filters, activation='relu', decoder_name = 'deconv', pretrained=True, output_size =(1024, 1024))

summary(model, (3, 1024, 512))    


model = nn.DataParallel(model)

if cuda:
    model.cuda()

cudnn.benchmark = True

###Loss function and optimizer
loss_criterion = nn.CrossEntropyLoss()
#loss_criterion = nn.BCEWithLogitsLoss()





# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)








def segtrain(train_dataloader, val_dataloader, model, loss_criterion, EPOCHS,  cuda=False, *args, **kwargs):

    print("===================================...==================================\n")
    print(str(datetime.now()).split('.')[0], "Starting training and validation...\n")
    print("================================Results...==============================\n")


    val_dataiter = iter(val_dataloader)
    train_iterator = iter(train_dataloader)
    model.train()
    pbar = tqdm(total=EPOCHS)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        val_running_loss = 0.0
        val_counter = 0
        train_counter = 0

        for i in range(0, len(train_dataloader)):
            # zero the parameter gradients
            optimizer.zero_grad()
            

            try:
                inputs, labels = next(train_iterator)
            except StopIteration:
                # now the dataset is empty -> new epoch
                train_iterator = iter(train_dataloader)
                inputs, labels = next(train_iterator)
                pbar.update(1)
            # get the inputs
            # if cuda:
                # print("Trying to push stuff do gpu")
            # inputs.cuda()
            # labels.cuda()

            
            
            # forward + backward + optimize
            if cuda==True:
                outputs = model(inputs.cuda())
                # measure loss
                loss = loss_criterion(outputs, labels.long().cuda())
                # loss = loss_criterion(outputs, labels.squeeze(1).long().cuda())

            else:
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels.long())



            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # update progress bar status  and print statistics
            running_loss += loss.item()
            train_counter += 1
            if i % LOG_FREQUENCY == 0:  # print every 100 mini-batches
                pbar.write('training epoch :%d at iteration %5d  with batch_loss: %f' %
                          (epoch + 1, i + 1, running_loss / train_counter))


            intersections = []
            unions = []


            #validation
            if (i + 1) % LOG_FREQUENCY == 0:
                torch.cuda.empty_cache()


                try:
                    val_images, val_targets = next(val_dataiter)
                except StopIteration: 
                    # now the dataset is empty -> new epoch
                    val_dataiter = iter(val_dataloader)
                    val_images, val_targets = next(val_dataiter)
                with torch.no_grad():
                    if cuda==True:
                        outputs = model(val_images.cuda())
                        loss = loss_criterion(outputs, val_targets.long().cuda()) 
                    else:
                        outputs = model(val_images)
                        loss = loss_criterion(outputs, val_targets.long()) 
                    val_running_loss += loss.item()
                    val_counter += 1

                    val_preds = torch.argmax(outputs.float(), dim=1).cpu()
                    #val_preds = torch.max(outputs.float(), dim=1).cpu()

                    compute_iou(val_targets.numpy(), val_preds.numpy(), CLASSES, intersections, unions)


                intersections = np.sum(intersections, axis = 0)
                unions = np.sum(unions, axis = 0)
                ious = [(intersections[k] + 1)/(unions[k] + 1) for k in range(len(intersections))]
                mIOU = np.nanmean(ious)
                print(ious)
                print('the mean IOU is ', mIOU)


            # # logging training/validation metrics and prediction images in Tensorboard
            # if (i + 1) % 500 == 0:
            #     # runs are organized by training start time
        try:
            val_images, val_targets = next(val_dataiter)
        except StopIteration: 
            # now the dataset is empty -> new epoch
            val_dataiter = iter(val_dataloader)
            val_images, val_targets = next(val_dataiter)
        
        writer = SummaryWriter(log_dir)
        
        writer.add_scalar('Validation loss', val_running_loss / val_counter, epoch)

        writer.add_scalar('Training loss', running_loss / train_counter, epoch)

        val_running_loss = 0.0
        running_loss = 0.0   
        train_counter = 0
        val_counter = 0

        writer.add_scalar('Mean_IOU', mIOU, epoch)

        # show performance on some images of the test set in tensorboard 
        writer.add_figure('Performance on test set example',
                visualize(model, val_images, inverse_class_converter, val_targets),global_step = epoch)
        writer.close()


        # saving weights of model after n iterations

        # if (i + 1) % 100 == 0: 
        #      #torch.save(model.state_dict(), 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_weights' +  current_time + '_' + str(i) + ".pth") #on computer
        #      torch.save(model.state_dict(), '/localdata/grace/Eze/Project/mtl_runs/train' +  current_time + '_' + str(i) + ".pth") #on server

        #if model == 'fcn8' and (i + 1) % 100 == 0 :
        #    torch.save(model.state_dict(), 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_weights' + current_time + '_' + 'fcn8' + '_' + str(i) + ".pth")

        #elif model == 'linknet' and (i + 1) % 100 == 0 :
        #    torch.save(model.state_dict(), 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_weights' + current_time + '_' + 'linknet8' + '_'+ str(i) + ".pth")



    ## save final model
    #torch.save(model.state_dict(), 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_weights' +  current_time + "_final.pth") #on computer
    torch.save(model.state_dict(), '/localdata/grace/Eze/Project/mtl_runs/train' +  current_time + "_final.pth")  #on server

    #if model == 'fcn8' :
    #    torch.save(model.state_dict(), 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_weights' + current_time + 'fcn8' + "_final.pth")

    #elif model == 'linknet':
    #    torch.save(model.state_dict(), 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_weights'   +  current_time + 'linknet8' + "_final.pth")

    pbar.close()





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











if __name__ == "__main__":
    

    segtrain(train_dataloader, val_dataloader, model, loss_criterion, EPOCHS, cuda=True)


