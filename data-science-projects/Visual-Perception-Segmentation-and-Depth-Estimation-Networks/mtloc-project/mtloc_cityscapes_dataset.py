from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, glob
import cv2
import torch
import numpy
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import time
import json
from collections import namedtuple
import zipfile
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset

from mtloc_convert_util import *
from mtloc_misc_utils import *
import matplotlib.pyplot as plt





"""
source: https://pytorch.org/docs/stable/_modules/torchvision/datasets/cityscapes.html

"""




#class CityscapesDepthSeg(VisionDataset):
class CityscapesDepthSeg(Dataset):

    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.


    Cityscapes dataset for Semantic, Instance Segmnentation ALSO! modified city scape dataloader for 
    depth estimation (disparity data from cityscapes)


    Args:
        root (string): Root directory of dataset where directory "leftImg8bit", "rightImg8bit"
            , "gtFine",  "gtCoarse" , "disparity" and "camera"  are located.
        split (string, optional): The image split to use, "train", "test" or "val" if mode="gtFine"
            otherwise "train", "train_extra" or "val"
        mode (string, optional): The quality mode to use, "gtFine" or "gtCoarse"
        target_type (string or list, optional): Type of target to use, "instance", "semantic", "polygon", "disparty" (and/or camera)
            or "color". Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, "transforms.RandomCrop"
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.


        disp_target_dir (str): Path to disparity directory
        cam_prop_dir (str, optional): Path to camera properties directory. Default: None
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        depth_scale=256: Default KITTI pre-process. divide 256 to get gt measured in meters (m)
        garg_crop=True: Following Adabins, use grag crop to eval results.
        eigen_crop=False: Another cropping setting.
        min_depth=1e-3: Default min depth value.
        max_depth=80: Default max depth value.

    split file format:
    input_image: leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png  
    gt_semantic: gtFine/train/cologne/aachen_000038_000019_gtFine_labelIds.png  
    gt_instance: gtFine/train/cologne/aachen_000038_000019_gtFine_instanceIds.png  
    gt_color:    gtFine/train/cologne/aachen_000038_000019_gtFine_color.png  
    gt_polygons: gtFine/train/cologne/aachen_000038_000019_gtFine_polygons.json 
    gt_depth:    disparity/train/aachen/aachen_000000_000019_disparity.png
    camera:       train/aachen/aachen_000000_000019_camera.json (disparity to depth)




    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]




    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(self, root, split='train', mode='fine', target_type='instance', is_cam='camera', transform=None, target_transform=None, augumentation= None, size=None, image_transform=None, depth_transform = None,
                  test_mode=False,  *args, **kwargs):
        #super(CityscapesDepthSeg, self).__init__(root, transform, image_transform,  target_transform)
        super(CityscapesDepthSeg, self).__init__(**kwargs)
        self.root=root

        self.mode = 'gtFine' if mode == 'fine' else ('disparity' if mode == 'disparity'  else 'gtCoarse' ) 
        self.is_cam = is_cam
        
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)

        self.disp_target_dir = os.path.join(self.root, 'disparity', split)
        self.cam_prop_dir = os.path.join(self.root, self.is_cam, split)
        self.target_type = target_type
        self.split = split
        self.augumentation = augumentation
        self.size = size
        self.images = []
        self.targets = []
        self.depth_targets = []
        self.camera_properties = []

        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform

        self.test_mode = test_mode




        verify_str_arg(mode, "mode", ("fine", "coarse", "disparity"))
        if mode == "fine" or self.target_type == "disparity":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)


        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color", "disparity"))
         for value in self.target_type]



        if not (os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir)) or not (os.path.isdir(self.disp_target_dir)):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))


            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))      

            if self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if self.mode == 'disparity' and self.is_cam is not None:
                disp_target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
                cam_prop_dir_zip = os.path.join(self.root, '{}{}'.format(self.is_cam, '_trainvaltest.zip'))

            if (self.mode == 'gtFine' and ('disparity' in self.target_type )) and self.is_cam is not None:
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))      
                disp_target_dir_zip = os.path.join(self.root, '{}{}'.format(self.target_type, '_trainvaltest.zip'))
                cam_prop_dir_zip = os.path.join(self.root, '{}{}'.format(self.is_cam, '_trainvaltest.zip'))



            
            if (self.mode == 'gtFine' or split == 'train_extra') or (self.mode == 'gtCoarse' and split == 'train_extra'):
                if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                    extract_archive(from_path=image_dir_zip, to_path=self.root)
                    extract_archive(from_path=target_dir_zip, to_path=self.root)

            if (self.mode == 'disparity' or split == 'train_extra') and (('disparity' in self.target_type ) and self.is_cam == 'camera'):

                if (os.path.isfile(image_dir_zip) and os.path.isfile(disp_target_dir_zip)) and (os.path.isfile(cam_prop_dir_zip)):
                    extract_archive(from_path=image_dir_zip, to_path=self.root)
                    extract_archive(from_path=disp_target_dir_zip, to_path=self.root)
                    extract_archive(from_path=cam_prop_dir_zip, to_path=self.root)

            if ((self.mode == 'gtFine' or split == 'train_extra') or (self.mode == 'gtCoarse' or split == 'train_extra')) and (('disparity' in self.target_type ) and self.is_cam == 'camera'):

                if (os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip)) and (os.path.isfile(disp_target_dir_zip) and os.path.isfile(cam_prop_dir_zip)):
                    extract_archive(from_path=image_dir_zip, to_path=self.root)
                    extract_archive(from_path=target_dir_zip, to_path=self.root)
                    extract_archive(from_path=disp_target_dir_zip, to_path=self.root)
                    extract_archive(from_path=cam_prop_dir_zip, to_path=self.root)

            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')


        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            disp_dir = os.path.join(self.disp_target_dir, city)
            cam_dir = os.path.join(self.cam_prop_dir, city)

            for file_name in os.listdir(img_dir):
                target_types = []
                disp_target_types = []
                cam_props = []
                for t in self.target_type:
                    

                    if t == 'disparity' and self.mode =='disparity':
                        disp_target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix('disparity', t))
                        cam_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.is_cam, 'camera'))

                    elif (t == 'disparity') and (self.mode == 'gtFine' or self.mode == 'gtCoarse'):
                        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, 'semantic'))
                        disp_target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix('disparity', t))
                        cam_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.is_cam, 'camera'))


                    else: 
                        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                               self._get_target_suffix(self.mode, t))
                        


                    if (t == 'disparity') and (self.mode == 'disparity'):
                        target_types.append(os.path.abspath(os.path.join(disp_dir, disp_target_name)))
                        cam_props.append(os.path.abspath(os.path.join(cam_dir, cam_name)))

                    elif (t == 'disparity') and (self.mode == 'gtFine' or self.mode == 'gtCoarse'):
                        target_types.append(os.path.abspath(os.path.join(target_dir, target_name)))
                        disp_target_types.append(os.path.abspath(os.path.join(disp_dir, disp_target_name)))
                        cam_props.append(os.path.abspath(os.path.join(cam_dir, cam_name)))

                    elif ((t == 'instance' or t == 'semantic') or (t == 'polygon' or t == 'color')) and (self.mode == 'gtFine' or self.mode == 'gtCoarse'):
                        target_types.append(os.path.abspath(os.path.join(target_dir, target_name)))



                self.images.append(os.path.join(img_dir, file_name))
                if (t == 'disparity') and (self.mode == 'disparity'):
                    self.depth_targets.append(disp_target_types)
                    self.camera_properties.append(cam_props)
                elif (t == 'disparity') and (self.mode == 'gtFine' or self.mode == 'gtCoarse'):
                    self.targets.append(target_types)
                    self.depth_targets.append(disp_target_types)
                    self.camera_properties.append(cam_props)
                elif ((t == 'instance' or t == 'semantic') or (t == 'polygon' or t == 'color')) and (self.mode == 'gtFine' or self.mode == 'gtCoarse'):
                    self.targets.append(target_types)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon" or "camera", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        depth_disp = []
        cam_vals = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            elif t == 'disparity' and (self.mode == 'gtFine' or self.mode == 'gtCoarse'):
                target = Image.open(self.targets[index][i]).convert('L')
                disp_target = Image.open(self.depth_targets[index][i]).convert('L')
                cam_pvals = self._load_json(self.camera_properties[index][i])
            else:
                target = Image.open(self.targets[index][i]).convert('L')

            targets.append(target)
            depth_disp.append(disp_target)
            cam_vals.append(cam_pvals)



        target = tuple(targets) if len(targets) > 1 else targets[0]
        disp_target = tuple(depth_disp) if len(depth_disp) > 1 else depth_disp[0]
        cam_values = tuple(cam_vals) if len(cam_vals) > 1 else cam_vals[0]

        if self.size is not None:

            image = image.resize(self.size)
            target = target.resize(self.size)
            disp_target = disp_target.resize(self.size)




        if self.augumentation is not None:
            #print('performing data augumentation ')
            image = self.augumentation(image)
            target = self.augumentation(target)
            disp_target = self.augumentation(disp_target)

        if self.image_transform is not None:
            image = self.image_transform(image)


        # if (self.target_transform is not None ) and ('disparity' in self.target_type ):
        #     target = self.target_transform(torch.from_numpy(np.array(target)))
        #     disp_target = torch.from_numpy(np.array(disp_target))
        
        # else:
        #     target = self.target_transform(torch.from_numpy(np.array(target)))

        if (self.target_transform is not None and self.depth_transform is not None) and ('disparity' in self.target_type ):
            target = self.target_transform(torch.from_numpy(np.array(target)))
            disp_target = self.depth_transform(torch.from_numpy(np.array(disp_target)))

        elif (self.depth_transform is  None) and ('disparity' in self.target_type ):
            disp_target = torch.from_numpy(np.array(disp_target))
    
            if self.target_transform is not None:
                target = self.target_transform(torch.from_numpy(np.array(target)))
            else:
                target = torch.from_numpy(np.array(target))
 
         
        else:

            if self.target_transform is not None:
                target = self.target_transform(torch.from_numpy(np.array(target)))
            else:
                target = torch.from_numpy(np.array(target))






        # Extrinsic and Intrinsic Parameters from camera file

        baseline = cam_values['extrinsic']['baseline'] 
        pitch = cam_values['extrinsic']['pitch']
        yaw = cam_values['extrinsic']['yaw']
        roll = cam_values['extrinsic']['roll']
        xp = cam_values['extrinsic']['x']
        yp = cam_values['extrinsic']['y'] 
        zp = cam_values['extrinsic']['z'] 
        focal_length = cam_values['intrinsic']['fx'] 
        fy = cam_values['intrinsic']['fy'] 
        up = cam_values['intrinsic']['u0'] 
        vp = cam_values['intrinsic']['v0'] 



        if ('disparity' in self.target_type ) and (self.mode == 'gtFine' or self.mode == 'gtCoarse'):
            #print('')
            return image, target, disp_target, baseline, focal_length, xp, yp, zp, up, vp, fy
        else:
            return image, target



    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'disparity':
            return '{}.png'.format(mode)
        else:
            return '{}.json'.format(mode)









if __name__ == '__main__':

    from itertools import chain

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    CLASSES = cityscapes_labelsWithTrainID
    target_transform = ClassConverter(CLASSES)
    #target_transform = ClassConverter([7, 9, 26, 27, 28])
    #target_transform = transforms.Compose([ClassConverter(CLASSES),
    #                                        transforms.ToTensor()])


    #depth_transform =  transforms.ToTensor()
    # depth_transform = transforms.Compose([DepthNorm(min_depth = 1e-3, max_depth = 80), 
    #                                         Disp2Depth(baseline, focal_length, depth_scale=256., scale=True)])

    augmentations = data_augmentations()

    ## Cityscapes for just semantic segmentation

    #transcityscape = CityscapesDepthSeg('C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes',
    #                            split='train',
    #                            mode='fine',
    #                            target_type='semantic',
    #                            augumentation=augmentations,
    #                            transform=None,
    #                            image_transform=transform,
    #                            size=(2048, 1024),
    #                            target_transform=target_transform)



    #val_cityscape = CityscapesDepthSeg('C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes',
    #                            split='val',
    #                            mode='fine',
    #                            target_type='semantic',
    #                            transform=None,
    #                            image_transform=transform,
    #                            size=(2048, 1024),
    #                            target_transform=target_transform)


    #train_dataloader = DataLoader(transcityscape, batch_size=3,
    #                             shuffle=True, num_workers=4, collate_fn=None)
    #val_dataloader = DataLoader(val_cityscape, batch_size=3,
    #                             shuffle=True, num_workers=4, collate_fn=None)


    #image, sem_label = next(iter(train_dataloader))
    #print(f"Image batch shape: {image.size()}")
    #print(f"semantic label batch shape: {sem_label.size()}")
    #img = image[0:1].squeeze()
    #img_wochannel = img[0].squeeze()
    #sem_label = sem_label[0]
    #plt.imshow(img_wochannel, cmap="gray")
    #plt.show()
    #print(f"semantic label: {sem_label}")



    # # Cityscapes for both semantic and depth 


    batch_size=1
    root_dir_local ='C:/Users/eze/Documents/PhD_project/datasets/mtloc_cityscapes' #local machine
    root_dir = '/localdata/grace/Eze/Datasets/cityscapes/mtloc_cityscapes' #server
    train_csds = CityscapesDepthSeg(root_dir,
                                split='train',
                                mode='fine',
                                target_type='disparity',
                                is_cam='camera',
                                transform = transform,
                                augumentation=None,
                                image_transform=transform,
                                depth_transform= None,
                                size=(2048, 1024),
                                target_transform=target_transform)


    val_csds = CityscapesDepthSeg(root_dir,
                                split='val',
                                mode='fine',
                                target_type='disparity',
                                is_cam='camera',
                                transform=transform,
                                image_transform=transform,
                                depth_transform=None,
                                size=(2048, 1024),
                                target_transform=target_transform)



    train_dsloader = DataLoader(train_csds, batch_size=batch_size,
                                 shuffle=True, num_workers=4, collate_fn=None)
    val_dsloader = DataLoader(val_csds, batch_size=batch_size,
                                 shuffle=True, num_workers=4, collate_fn=None)






    #for unpacking values for disparity to depth training
    image, sem_label, disp_label, baseline, focal_length, xp, yp, zp, up, vp, fy = next(iter(train_dsloader))

    print(f"Image batch shape: {image.size()}")
    print(f"semantic label batch shape: {sem_label.size()}")
    print(f"disparity label batch shape: {disp_label.size()}")
    print(f"baseline: {baseline}")
    print(f"focal length: {focal_length}")
    print(f"xp : {xp}")
    print(f"yp: {yp}")
    print(f"zp: {zp}")
    print(f"up: {up}")
    print(f"vp: {vp}")
    print(f"fy: {fy}")
    print(f'baseline shape: {baseline.shape}')
    img = image[0].squeeze()
    imgs = img[0].squeeze()
    s_label = sem_label[0]
    d_label = disp_label[0]
    plt.imshow(imgs, cmap="gray")
    plt.show()

    to_depth_coverter = transforms.Compose([Disp2Depth(baseline, focal_length, depth_scale=256, batch_size=batch_size, scale=True),
                                                    DepthNorm(min_depth = 1e-3, max_depth = 126, batch_size=batch_size)])       


    # to_depth_coverter = Disp2DepthNorm(baseline, focal_length, depth_scale=256, min_depth = 1e-3, max_depth = 126, batch_size=batch_size, scale=True, mimax=True)
                                                         



    depth_label = to_depth_coverter(disp_label)
    de_label = depth_label[0]

    print(f"semantic label: {sem_label}")
    print(f"disparity label: {disp_label}")
    print(f"depth label: {depth_label}")

    print(f"disparity label pixel i: {d_label[712][756]}")
    print(f"depth label pixel i: {de_label[712][756]}")
    print(f"depth label pixel j: {de_label[912][1256]}")


    # disp_uniq_vals, idx, counts = np.unique(de_label, axis=0, return_index=True, return_counts=True)
    # print ("Unique values in matrix are : " , disp_uniq_vals)
    # print('count', counts)











