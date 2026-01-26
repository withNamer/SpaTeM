import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from networks.vnet import VNet
from networks.unet2D import UNet
from utils import ramps, losses
from dataloaders.la_heart import TwoStreamBatchSampler
from dataloaders.ACDC import ACDCDataset
# from functools import reduce
# from test_util import test_all_case_array
from utils.lr_scheduler import PolyLR
from datetime import datetime
from skimage.measure import label
from medpy import metric

# 这个很可能需要加上largest的技术

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/data/ldap_shared/home/***/PMT/data/ACDC",
    help="Name of Experiment",
)  
parser.add_argument("--exp", type=str, default="PMT_B_ACDC", help="model_name") 
parser.add_argument(
    "--max_iterations", type=int, default=80000, help="maximum epoch number to train"
)  
parser.add_argument(
    "--batch_size", type=int, default=24, help="batch_size per gpu"
) 
parser.add_argument(
    "--labeled_bs", type=int, default=12, help="labeled_batch_size per gpu"
) 
parser.add_argument(
    "--base_lr", type=float, default=0.06, help="maximum epoch number to train"
) 
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)  
parser.add_argument("--seed", type=int, default=1337, help="random seed")  
parser.add_argument("--gpu", type=str, default="3", help="GPU to use")  
parser.add_argument(
    "--ema_decay", type=float, default=0.999, help="ema_decay"
)  
parser.add_argument(
    "--consistency_type", type=str, default="mse", help="consistency_type"
)  
parser.add_argument(
    "--consistency", type=float, default=20.0, help="consistency"
)  
parser.add_argument(
    "--consistency_rampup", type=float, default=20.0, help="consistency_rampup"
)  
parser.add_argument("--model_num", type=int, default=2, help="model_num")
parser.add_argument("--epoch_step", type=int, default=324, help="epoch step") # 1312
args = parser.parse_args() 

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  
num_classes = 2

def get_acdc_2d_cct(segmentation):
    batch_list = []
    mask = torch.argmax(segmentation, dim=1).detach().cpu().numpy()
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = mask[i]
            labels = label(temp_seg == c)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                if largestCC.sum() < 10:
                    class_list.append(largestCC * 0)
                else:
                    class_list.append(largestCC * c)
            else:
                class_list.append(labels)
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)
    batch_list = torch.tensor(batch_list).cuda()
    if len(torch.unique(batch_list)) == 1: return segmentation
    return torch.nn.functional.one_hot(batch_list, num_classes=4).permute(0, 3, 1, 2) * segmentation

def test_single_case(net_array, image, cct=False):
    from scipy.ndimage.interpolation import zoom
    volume_pred = np.zeros_like(image)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        out_main = 0
        for net in net_array:
            out_main += net(torch.tensor(slice, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(1)) 
        out_main = out_main / len(net_array)
        out_main = get_acdc_2d_cct(torch.tensor(out_main)) if cct else out_main
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0)
        volume_pred[ind] = pred
    return volume_pred

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

val_set = ACDCDataset("/ldap_shared/home/***/PMT/data/ACDC",
                              "/ldap_shared/home/***/PMT/data/ACDC",
                              split="test")


def create_model(name):
    # Network definition
    if name == "unet":
        net  = UNet(in_chns=1, class_num=4)
        model = net.cuda()
    return model



def test_calculate_metric(epoch_num):
    model_array = []

    model_save_path1 = ''
    model1 = create_model(name="unet")

    model1.eval()
    model1.load_state_dict(torch.load(model_save_path1))
    model_array.append(model1) 

    model_save_path2 = ''
    model2 = create_model(name="unet")
    model2.eval()
    model2.load_state_dict(torch.load(model_save_path2))
    model_array.append(model2)

    dataloader = iter(val_set)
    tbar = range(len(val_set))
    tbar = tqdm(tbar, ncols=135)
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for batch_idx in tbar:
        x, y = next(dataloader)
        y_tilde = test_single_case(model_array, x)
        if np.sum(y_tilde == 1) == 0:
            first_metric = 0, 0, 0, 0
        else:
            first_metric = calculate_metric_percase(y_tilde == 1, y == 1)
        if np.sum(y_tilde == 2) == 0:
            second_metric = 0, 0, 0, 0
        else:
            second_metric = calculate_metric_percase(y_tilde == 2, y == 2)

        if np.sum(y_tilde == 3) == 0:
            third_metric = 0, 0, 0, 0
        else:
            third_metric = calculate_metric_percase(y_tilde == 3, y == 3)

        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    metric_record = (first_total + second_total + third_total) / (3 * len(val_set))
    print("metric_record: ", metric_record)

    return metric_record


if __name__ == "__main__":
    iters = 24000
    metric = test_calculate_metric(iters)
    # print("iter:", iter)
    print(metric)
