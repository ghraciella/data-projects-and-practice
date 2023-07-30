import torch
import torch.nn as nn
import math
import random
import numpy as np
from torchmetrics.metric import Metric
from torch import Tensor, tensor

from mtloc_cityscapes_dataset import *






#Computing the intersection over union

def compute_iou(gt_true, gt_pred, CLASSES, inter_matrix, union_matrix):

    ious =  []
    intersections = []
    unions = []
    for k in range(0, len(CLASSES)+1):

        tp = np.sum((gt_true==k)&(gt_pred==k))
        fp = np.sum((gt_true==k)|(gt_pred==k))
        # fn = np.sum((gt_true==k)&(gt_pred!=k))
        intersections.append(tp)
        unions.append(fp)


    inter_matrix.append(intersections)
    union_matrix.append(unions)





#evaluaation metrics for depth

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10



def prop_3d_metric(gt, pred, min_depth=1e-3, max_depth=80, *args, **kwargs):
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    


    return x_hat



def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


def _check_for_empty_tensors(preds: Tensor, target: Tensor) -> bool:
    if preds.numel() == target.numel() == 0:
        return True
    return False


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )

# def compute_error(regression_error, n_observations):

#     return regression_error/n_observations




class Evaluation(Metric):

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    mse: Tensor
    rmse: Tensor
    mae: Tensor
    irmse: Tensor
    imae: Tensor
    absrel: Tensor
    log10: Tensor
    delta1: Tensor
    delta2: Tensor
    delta3: Tensor
    #data_time: Tensor
    #gpu_time: Tensor
    total: Tensor    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("mse", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("rmse", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("mae", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("irmse", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("imae", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("absrel", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("log10", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("delta1", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("delta2", default=tensor(0.0), dist_reduce_fx="sum")

        self.add_state("delta3", default=tensor(0.0), dist_reduce_fx="sum")

        #self.add_state("data_time", default=tensor(0.0), dist_reduce_fx="sum")
        
        #self.add_state("gpu_time", default=tensor(0.0), dist_reduce_fx="sum")
    
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")



    def update(self, predictions, targets): 
        "update state with predictions and targets" 

        n_obs = targets.numel()


        #print('prediction shape', predictions.shape)

        #print('targets', targets.shape)




        #rmse_log = (torch.log(targets) - torch.log(predictions)) ** 2
        #rmse_log = torch.sqrt(rmse_log.mean())
        #sq_rel = torch.mean((targets - predictions) ** 2 / targets)

        abs_diff = (targets - predictions).abs()
        mse = float((torch.pow(abs_diff, 2)).mean())
        rmse = torch.sqrt(self.mse)
        mae = float(abs_diff.mean())
        lg10 = float((torch.log(targets) - torch.log(predictions)).abs().mean())
        absrel = float((abs_diff / targets).mean())



        maxRatio = torch.max((targets/predictions), (predictions/targets))
        delta1 = float((maxRatio < 1.25).float().mean())
        delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        #data_time = 0
        #gpu_time = 0

        inv_output = 1 / predictions
        inv_target = 1 / targets
        abs_inv_diff = (inv_target - inv_output).abs()
        irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        imae = float(abs_inv_diff.mean())


        self.mse += mse
        self.rmse += rmse
        self.mae += mae
        self.irmse += irmse
        self.imae += imae
        self.absrel += absrel
        self.log10 += lg10
        self.delta1 += delta1
        self.delta2 += delta2
        self.delta3 += delta3
        #self.data_time += data_time
        #self.gpu_time += gpu_time

        self.total += n_obs

    def compute(self):
        """Computes error over state."""

        mse_state = self.mse/self.total
        rmse_state = self.rmse/self.total
        mae_state = self.mae/self.total
        irmse_state = self.irmse/self.total
        imae_state = self.imae/self.total
        absrel_state = self.absrel/self.total
        log10_state = self.log10/self.total
        delta1_state = self.delta1/self.total
        delta2_state = self.delta2/self.total
        delta3_state = self.delta3/self.total
        #data_time_state = self.data_time/self.total
        #gpu_time_state = self.gpu_time/self.total


        return mse_state, rmse_state, mae_state,irmse_state, imae_state, absrel_state, log10_state, delta1_state, delta2_state, delta3_state #, data_time_state, gpu_time_state










if __name__ == "__main__":

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



    #CLASSES = cityscapes_labelsWithTrainID

    #transform = transforms.ToTensor()
    #target_transform = ClassConverter(CLASSES)
    #inverse_class_converter = target_transform.get_inverter()

    #final_model_weights = '/localdata/Grace/Codes/model_weights/04-02-2020_21-48-06_final.pth'


    #model = FCN8(len(CLASSES) + 1)
    #model.load_state_dict(torch.load(final_model_weights))
    #model.eval()
    #model.cuda()

    # image = "/localdata/Datasets/cityscapes/leftImg8bit/train/bremen/bremen_000000_000019_leftImg8bit.png"
    # testgtimage = "/localdata/Datasets/cityscapes/gtFine_trainvaltest/gtFine/train/bremen/bremen_000000_000019_gtFine_labelIds.png"
    # gt_true = cv2.imread(testgtimage, cv2.IMREAD_GRAYSCALE)
    # gttrue = cv2.resize(gt_true, (1024,512))

    # with torch.no_grad():
    #     image = cv2.imread(image)
    #     image = cv2.resize(image, (1024,512))
    #     gt_pred = model(transform(image).unsqueeze(0).cuda())


    # gtpred = torch.argmax(gt_pred.float(), dim=1).cpu().numpy()[0]
    
    # out = visualizeMask(gtpred, inverse_class_converter)
    # gttrue = target_transform(gttrue).numpy()
    # #cv2.imshow('test', out)
    # #key = cv2.waitKey(0)

    # #print(gttrue.shape, gtpred.shape)
    # intersections = []
    # unions = []
    # # for image, gt in dataset
    #     # out = net(image)
    #     # compute_iou(gt, out, CLASSES, intersections, unions)
    # compute_iou(gttrue, gtpred, CLASSES, intersections, unions)
    # intersections = np.sum(intersections, axis = 0)
    # unions = np.sum(unions, axis = 0)
    # ious = [(intersections[k] + 1)/(unions[k] + 1) for k in range(len(intersections))]
    # mIOU = np.nanmean(ious)
    # print(ious)
    # print('the mean IOU is ', mIOU)




    cv2.waitKey(0)








