
import sys, os

sys.path.append("..")

import collections.abc
from collections.abc import MutableMapping
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor


from mtloc_cityscapes_dataset import *
from mtloc_convert_util import *
from mtloc_metrics import *
from mtloc_misc_utils import *
from mtloc_models import *
#from mtloc_mtltrain import *
from mtloc_objective_losses import *
from mtloc_optimizers import *
from mtloc_PLtrain import *
from mtloc_visualise import *

os.environ["WANDB_SILENT"] = "true"

#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'




def pl_main():

    pl.seed_everything(1) #seed_everything(42, workers=True)

    EPOCHS = 50


    ##logging training
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    #log_dir = 'C:/Users/eze/Documents/PhD_project/mtloc_runs_weights/mtloc_runs/train' + current_time #local computer
    log_dir = "/localdata/grace/Eze/Project/mtl_runs/train" #remote server 
    tb_log_dir = "/localdata/grace/Eze/Project/mtl_runs/tensorboard_logs" #remote server 
    wdb_log_dir = "/localdata/grace/Eze/Project/mtl_runs/wandb_logs/lightning_logs" #remote server 
    ckpt_log_dir = "/localdata/grace/Eze/Project/mtl_runs/modelcheckpoint/lightning_logs" #remote server
    lightning_logs = "/localdata/grace/Eze/Project/mtl_runs/lightning_logs" #remote server 
    model_file_name = 'Disp_DeNet' #+ current_time

    # tb_logger = pl.loggers.TensorBoardLogger('Tensorboard')
    wandb_logger = pl.loggers.WandbLogger(save_dir=wdb_log_dir, project='DeNet', name='Denet_train', log_model=True)


    #net = DisparityNet(in_channels = 3, num_class=1, filters=filters, activation='relu', decoder_name = 'deconv', pretrained=True, output_size = (1024, 1024))

    #wandb_logger.watch(net, log="all", log_freq=200, log_graph=True) #track model's gradient and topology
    #wandb_logger.experiment.unwatch(net)




    class PrintCallbacks(pl.Callback):

        def on_init_start(self, trainer):
            print('Starting to initialize trainer')

        def on_init_end(self, trainer):
            print('Trainer is initialized now')
            
        def on_train_end(self, trainer, pl_module):
            print('Training ended succesfully')


   



    checkpoint_callback = ModelCheckpoint(
         dirpath= ckpt_log_dir,
         filename=model_file_name + '{epoch}-{val_loss:.2f}-{other_metric:.2f}',
         monitor = 'val_loss',
         mode='min',
         save_top_k = 1,
         save_weights_only = False)

    #profiler = AdvancedProfiler()
    callbacks = [checkpoint_callback, 
                ModelSummary(max_depth=1), 
                LearningRateMonitor(logging_interval="step") ]

    tb_logger = pl.loggers.TensorBoardLogger('Tensorboard')
    



    data_tv = Data()
    task = DeNet(target_name='depth', optimizer_name= 'Adam', loss_name= 'Others')
    train_data = data_tv.train_dataloader()
    val_data = data_tv.val_dataloader()
    trainer = pl.Trainer(max_epochs = EPOCHS, accelerator= 'gpu')

    # trainer = pl.Trainer(accelerator= 'gpu', #'auto',
    #                         devices=3 if torch.cuda.is_available() else None,                           
    #                         default_root_dir=lightning_logs, 
    #                         logger=  [tb_logger, wandb_logger], #wandb_logger, #[tb_logger, wandb_logger],
    #                         max_epochs=EPOCHS,
    #                         check_val_every_n_epoch=1,
    #                         enable_progress_bar=False,
    #                         deterministic=True,
    #                         enable_model_summary=True,
    #                         sync_batchnorm=True)


    trainer.fit(task, train_data, val_data)
    #trainer.save_checkpoint("best_model.ckpt" )

    #trainer.validate(datamodule=data_tv)

    wandb.finish()







if __name__ == "__main__":
    


    pl_main()


    # # use model after training or load weights and drop into the production system
    # model = DeNet.load_from_checkpoint("best_model.ckpt")
    # x = ...
    # model.eval()
    # with torch.no_grad():
    #     y_hat = model(x)

















