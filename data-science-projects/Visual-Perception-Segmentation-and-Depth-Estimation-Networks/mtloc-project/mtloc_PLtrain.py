import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from datetime import datetime


from mtloc_cityscapes_dataset import *
from mtloc_convert_util import *
from mtloc_models import *
from mtloc_misc_utils import *
from mtloc_metrics import *
from mtloc_optimizers import *
from mtloc_objective_losses import *
from mtloc_visualise import *
from mtloc_mtltrain import *

import wandb



current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

run = wandb.init(project="DeNet", entity="ghraciella", name='train_metrics'+current_time, sync_tensorboard=True)

#wandb.define_metric("epoch")

#log_dir = 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_runs/train' + current_time
log_dir = '/localdata/grace/Eze/Project/mtl_runs/train' + current_time  #remote server 

wdb_log_dir = "/localdata/grace/Eze/Project/mtl_runs/wandb_logs/lightning_logs" #remote server 

wandb_logger = pl.loggers.WandbLogger(save_dir=wdb_log_dir, project='DeNet', name='Denet_train' , log_model=True)


class Data(pl.LightningDataModule):
    def __init__(self):
        super(Data, self).__init__() 


        self.data_size = (1024, 1024)   
        self.batch_size = 3 if torch.cuda.is_available() else 1

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])

        self.transform = transforms.ToTensor()                                 

        #self.transform = transforms.Compose([ DepthNorm(min_depth = 1e-3, max_depth = 80) ])

        #self.depth_transform = DepthNorm(min_depth = 10, max_depth = 80)
                                
                                 
        #self.target_transform = ClassConverter(CLASSES)
        self.augmentations = data_augmentations()
        #self.inverse_class_converter = self.target_transform.get_inverter()

        #self.root_dir = 'C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes' #on local computer
        self.root_dir = '/localdata/grace/Eze/Datasets/cityscapes/mtloc_cityscapes' #on remote server (karush)


        # self.transcityscape = CityscapesDepthSeg(self.root_dir,
        #                         split='train',
        #                         mode='fine',
        #                         target_type='semantic',
        #                         augumentation=self.augmentations,
        #                         transform=None,
        #                         image_transform=self.transform,
        #                         #depth_transform=None,
        #                         size=self.data_size,
        #                         target_transform=self.target_transform)


        # self.val_cityscape = CityscapesDepthSeg(self.root_dir,
        #                         split='val',
        #                         mode='fine',
        #                         target_type='semantic',
        #                         transform=None,
        #                         image_transform=self.transform,
        #                         #depth_transform=None,
        #                         size=self.data_size,
        #                         target_transform=self.target_transform)


        self.train_csds = CityscapesDepthSeg(self.root_dir,
                                    split='train',
                                    mode='fine',
                                    target_type='disparity',
                                    is_cam='camera',
                                    augumentation=None,
                                    image_transform=self.transform,
                                    depth_transform=None,
                                    size=self.data_size,
                                    target_transform=None)


        self.val_csds = CityscapesDepthSeg(self.root_dir,
                                    split='val',
                                    mode='fine',
                                    target_type='disparity',
                                    is_cam='camera',
                                    image_transform=self.transform,
                                    depth_transform=None,
                                    size=self.data_size,
                                    target_transform=None)



    def train_dataloader(self):
        # train_dataloader = DataLoader(self.transcityscape, batch_size=self.batch_size,
        #                          shuffle=True, num_workers=4, persistent_workers=True, collate_fn=None)

        train_dsloader = DataLoader(self.train_csds, batch_size=self.batch_size,
                                     shuffle=True, num_workers=4, persistent_workers=True, collate_fn=None)

        return train_dsloader
        

    def val_dataloader(self):

        # val_dataloader = DataLoader(self.val_cityscape, batch_size=self.batch_size,
        #                          shuffle=False, num_workers=4, persistent_workers=True, collate_fn=None)

        val_dsloader = DataLoader(self.val_csds, batch_size=self.batch_size,
                                     shuffle=False, num_workers=4, persistent_workers=True, collate_fn=None)

        return val_dsloader
        

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass




class DeNet(pl.LightningModule):
    def __init__(self, target_name, optimizer_name, loss_name):
        super(DeNet, self).__init__()

        self.batch_size = 3 if torch.cuda.is_available() else 1
        self.lr = 0.001
        self.filters = [16, 32, 64, 128, 256, 512]
        self.LOG_FREQUENCY = 100
        self.segCLASSES = cityscapes_labelsWithTrainID
        self.num_of_classes = len(self.segCLASSES) + 1 #for segmentation
        self.target_transform = ClassConverter(self.segCLASSES)
        #self.target_transform = ClassConverter([7, 9, 26, 27, 28])       
        self.inverse_class_converter = self.target_transform.get_inverter()
        
        
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.save_hyperparameters()
        self.output_size = (1024, 1024)   


        #self.model_name = model_name
        self.model = DisparityNet(in_channels = 3, num_class=1, filters=self.filters, activation='relu', decoder_name = 'deconv', pretrained=True, output_size = self.output_size)

        self.optimizer_name = optimizer_name
        self.target_name = target_name

        self.loss_name = loss_name
        self.loss_criterion = nn.HuberLoss()
        self.comb_loss_criterion = CombDepthLoss()
        self.seg_loss_criterion =  nn.CrossEntropyLoss()


        self.t0_time = time.time()
        self.tf_time = time.time()     


        #self.regression_eval = Evaluation()
        self.val_regression_eval = Evaluation()


        #wandb.watch(self.model, log="all", log_freq=200, log_graph=True) #track model's gradient and topology


    def forward(self, out):

        output = self.model(out)        
        return output



    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]



    def loss_function(self, predictions, targets):

        if self.loss_name == 'Others':
            error = self.loss_criterion(predictions, targets)
        if self.loss_name == 'Comb':
            error = self.comb_loss_criterion(predictions, targets)
        if self.loss_name == 'segment':
            error = self.seg_loss_criterion(predictions, targets)

        return error 



    def training_step(self, train_batch, batch_idx):
        

        data_time = time.time() - self.t0_time

        if self.target_name == 'depth':
            train_imgs, train_sem_labels, train_disp_labels, baseline, focal_length, xp, yp, zp, up, vp, fy = train_batch
        else:
            train_imgs, train_disp_labels = train_batch



     

        #to_depth_coverter = Disp2DepthNorm(baseline, focal_length, depth_scale=256, min_depth = 1e-3, max_depth = 126, batch_size=self.batch_size, scale=True, norm=True)

        to_depth_coverter = transforms.Compose([Disp2Depth(baseline, focal_length, depth_scale=256, batch_size=self.batch_size, scale=False)])

        depth_map_label =  to_depth_coverter(train_disp_labels)

        #depth_map_label = train_disp_labels
        depth_map_label = depth_map_label.type(torch.float32)


        #segmentation

        #sem_labels = to_seg_converter()

        data_time = time.time() - self.t0_time
        
        self.tf_time = time.time()            

        train_preds = self.forward(train_imgs)

        train_preds = to_depth_coverter(train_preds)

        loss = self.loss_function(train_preds, depth_map_label)



        gpu_time = time.time() - self.tf_time

        #self.regression_eval.update(train_preds, depth_map_label)


        #log_metrics = {'train_loss': loss, 'data_time': data_time, 'gpu_time': gpu_time}

        #run.log(log_metrics)#, commit=False)        

        # for k,v in log_metrics.items():
        #     self.log(k,v, sync_dist=True, prog_bar=True, logger=True)

        return {'loss': loss} 


    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #run.log({'avg_train_loss': avg_train_loss})
        self.log('avg_train_loss', avg_train_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        # for k,v in log_metrics.items():
        #     self.log(k,v, sync_dist=True, prog_bar=True, logger=True)


    def validation_step(self, valid_batch, batch_idx):


        val_data_time = time.time() - self.t0_time


        if self.target_name == 'depth':
            val_imgs, val_sem_labels, val_disp_labels, val_baseline, val_focal_length, val_xp, val_yp, val_zp, val_up, val_vp, val_fy = valid_batch
        else:
            val_imgs, val_disp_labels = valid_batch


     

        #to_depth_coverter = Disp2DepthNorm(val_baseline, val_focal_length, depth_scale=256, min_depth = 1e-3, max_depth = 126, batch_size=self.batch_size, scale=True, norm=True)

        to_depth_coverter = transforms.Compose([Disp2Depth(val_baseline, val_focal_length, depth_scale=256, batch_size=self.batch_size, scale=True)])



        #val_depth_map_label = val_disp_labels


        val_depth_map_label =  to_depth_coverter(val_disp_labels)


        val_depth_map_label = val_depth_map_label.type(torch.float32)

        val_data_time = time.time() - self.t0_time
        
        self.tf_time = time.time()            

        val_preds = self.forward(val_imgs)

        val_preds = to_depth_coverter(val_preds)

            
        loss = self.loss_criterion(val_preds, val_depth_map_label)
        self.val_regression_eval.update(val_preds.detach(), val_depth_map_label.detach())


        val_gpu_time = time.time() - self.tf_time



        #val_log_metrics = {'val_loss': loss, 'val_data_time': val_data_time, 'val_gpu_time': val_gpu_time}
        
        #run.log(val_log_metrics)#, commit=False)
         


        if batch_idx % 100: # Log every 100 batches
            torch.cuda.empty_cache()
            self._log_image_predictions(val_imgs, val_preds, val_depth_map_label) 


            # writer = SummaryWriter(log_dir)
            
            # # show performance on some images of the test set in tensorboard 
            # writer.add_figure('Performance on test set example',
            #         visualize(model, val_images, inverse_class_converter, val_targets),global_step = epoch)
            # writer.close()


        return {'loss': loss} 


    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()


        v_mse, v_rmse, v_mae, v_irmse, v_imae, v_absrel, v_log10, v_delta1, v_delta2, v_delta3 = self.val_regression_eval.compute()

        val_log_metrics = {'avg_val_loss': avg_val_loss, 'val mse': v_mse, 
                        'val rmse': v_rmse, 'val mae': v_mae, 'val irmse': v_irmse,
                        'val imae': v_imae, 'val absrel': v_absrel, 'val log10': v_log10,
                        'val delta1': v_delta1, 'val delta2': v_delta2, 'val delta3': v_delta3}
        

        #run.log(val_log_metrics)#, commit=False) 

        #self.log_dict(val_log_metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)       
        for k,v in val_log_metrics.items():
            self.log(k,v, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
            #wandb.log(k,v, sync_dist=True, prog_bar=True)


        self.val_regression_eval.reset()
     
    
    
    def _log_image_predictions(self, image, target, prediction):


        columns = ['image', 'ground truth', 'prediction']

        target = target.squeeze()
        prediction = prediction#.squeeze()

        # print(targets)
        # print(predictions)
        # print(image.shape)
        # print(target.shape)
        # print(prediction.shape)
        # print(type(targets))
        # print(type(predictions))

 

        # print('len im', len(image))
        # print('len pre', len(prediction))
        # print('len tar', len(target))

        #colorize(targ, cmap_name='viridis', vmin=None, vmax=None)        

        data = [[wandb.Image(img), wandb.Image(colorize(targ, cmap_name='viridis')), 
                    wandb.Image(colorize(pred, cmap_name='viridis'))] for img, targ, pred in list(zip(image, target, prediction))]
        wandb_logger.log_table(key='data visuals table', columns=columns, data=data)
        wandb.finish()






#wandb.finish(quiet=True)



























        



















