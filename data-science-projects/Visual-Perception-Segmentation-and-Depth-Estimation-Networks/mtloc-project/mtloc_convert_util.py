import cv2
import numpy as np
import torchvision.transforms as transforms
from collections import namedtuple
import torch

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (165,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

cityscapes_propabilities = [1.12990371e-04, 4.58409167e-02, 1.30404310e-02, 1.50842667e-02,
 1.34239003e-02, 2.85600678e-03, 1.21220385e-02, 3.26399687e-01,
 5.38691021e-02, 6.26141364e-03, 1.80143693e-03, 2.02056519e-01,
 5.80210614e-03, 7.76630145e-03, 8.77063014e-05, 2.86265413e-03,
 5.38998291e-04, 1.08653968e-02, 8.01201828e-05, 1.83956371e-03,
 4.88027893e-03, 1.41013007e-01, 1.02499210e-02, 3.55792079e-02,
 1.07911733e-02, 1.19620604e-03, 6.19212377e-02, 2.36772938e-03,
 2.08210184e-03, 3.99641630e-04, 2.08457979e-04, 2.06185269e-03,
 8.73397699e-04, 3.66423038e-03]

cityscapes_extended_propabilities = [5.77653494e-02, 1.36718523e-02, 3.88925136e-03, 4.49881637e-03,
 4.00361938e-03, 8.51791497e-04, 3.61534482e-03, 1.63629361e-01,
 2.23410975e-02, 1.86743916e-03, 5.37270663e-04, 1.01153135e-01,
 3.20683290e-03, 5.49520870e-03, 2.61580197e-05, 8.53774040e-04,
 1.60753876e-04, 6.09160387e-03, 2.38954931e-05, 1.10314080e-03,
 2.50242028e-03, 8.27849044e-02, 6.23545520e-03, 6.39605369e-02,
 3.99283330e-03, 4.20400433e-04, 4.34905616e-02, 3.70083912e-03,
 2.33549314e-03, 1.19191363e-04, 6.21716779e-05, 6.57289835e-04,
 3.34770901e-04, 1.25120727e-03]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# label id to color dict
cityscapes_label2color = { label.id : label.color for label in labels }
cityscapes_label2name = { label.id : label.name for label in labels }

cityscapes_labelsWithTrainID = [label.id for label in labels if label.trainId in range(19)]



class ClassConverter(object):
    def __init__(self, classes, test = False):
        self.classes = classes
        self.inverter = dict()
        self.inverter[0] = 0
        self.test = test
        for k, classId in enumerate(self.classes):
            self.inverter[k + 1] = classId

    def __call__(self, target):
        height, width = target.shape
        newtarget = np.zeros((height, width)).astype(np.int64)
        for k, classId in enumerate(self.classes):
            newtarget[target == classId] = k + 1
        if self.test:
            return newtarget.astype(np.uint8)
        return torch.from_numpy(newtarget)
    
    def invert_(self, target):
        height, width = target.shape
        newtarget = np.zeros((height, width)).astype(np.uint8)
        for k in range(len(self.classes)):
            newtarget[target == k + 1] = self.classes[k]
        return newtarget.astype(np.uint8)

    def get_inverter(self):
        return self.inverter


def get_color_converter(classes):
    converter = dict()
    converter[0] = 0
    for k, classId in enumerate(classes):
        converter[k + 1] = classId
    return converter


def convertToClasses(mask, classes):
    """ This function takes a label-id ground truth mask and converts it to a mask only containing the labels specified in the classes vector.
        A conversion dict that gives the original id for the 'new' classes is also returned.
    """
    height, width = mask.shape[:2]
    newmask = np.zeros((height, width)).astype(np.uint8)
    for k, classId in enumerate(classes):
        newmask[mask == classId] = k + 1
    return newmask

def visualizeMask(mask, IdConverter, rgbimage = None, flip = False):
    """ This function visualizes the in classes specified classes for a ground truth mask mask.
        If the original rgb image is also supplied the hue and saturation of the mask will be applied to the rgb image
    """
    height, width = mask.shape
    classes = len(IdConverter.keys())
    outputimage = np.zeros((height, width, 3)).astype(np.uint8)
    #if not flip:
    #    for classId in range(classes):
    #        outputimage[mask == classId] = cityscapes_label2color[IdConverter[classId]] 
    #else:
    #    for classId in range(classes):
    #        outputimage[mask == classId] = tuple(reversed(cityscapes_label2color[IdConverter[classId]]))

    for classId in range(classes):
        outputimage[mask == classId] = cityscapes_label2color[IdConverter[classId]] 


    if rgbimage is not None:
        outputimage = cv2.cvtColor(outputimage, cv2.COLOR_RGB2HSV)
        #print('shape of rgb image', rgbimage.shape)
        rgbimage = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2HSV)
        rgbimage[:,:,:2] = outputimage[:,:,:2]
        rgbimage[:,:,2] = rgbimage[:,:,2] // 2 + 128
        outputimage = cv2.cvtColor(rgbimage, cv2.COLOR_HSV2RGB)
        #cv2.imshow("mask", outputimage)
        #cv2.waitKey(0)
        #cv2.imwrite("test.png", rgbimage)
    return outputimage
        

if __name__ == "__main__":
    testimage = "bremen_000004_000019"
    testImageRGB = cv2.imread(testimage + "_leftImg8bit.png")
    testGT = cv2.imread(testimage + "_gtFine_labelIds.png", cv2.IMREAD_GRAYSCALE)
    #cv2.namedWindow('rgbleftbit', cv2.WINDOW_NORMAL)
    #cv2.imshow("rgbleftbit",testImageRGB)
    #cv2.imshow("GT",testGT)
    classes = range(34) #[7] # labelsWithTrainID
    color_converter = get_color_converter(classes)
    class_converter = ClassConverter(classes)
    newmask = class_converter(testGT)
    #visualizeMask(newmask, color_converter, testImageRGB, len(classes))
    outimage = visualizeMask(newmask, color_converter, testImageRGB, len(classes))
    cv2.namedWindow('outimage', cv2.WINDOW_NORMAL)
    cv2.imshow("outimage",outimage)
    cv2.waitKey(0)









