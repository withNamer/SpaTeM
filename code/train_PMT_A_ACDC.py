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
from utils.dice import DiceLoss

# 这个很可能需要加上largest的技术

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/data/ldap_shared/home/***/PMT/data/ACDC",
    help="Name of Experiment",
)  
parser.add_argument("--exp", type=str, default="PMT_A_ACDC", help="model_name") 
parser.add_argument(
    "--max_iterations", type=int, default=48000, help="maximum epoch number to train"
)  
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch_size per gpu"
) 
parser.add_argument(
    "--labeled_bs", type=int, default=16, help="labeled_batch_size per gpu"
) 
parser.add_argument(
    "--base_lr", type=float, default=0.068, help="maximum epoch number to train"
) 
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)  
parser.add_argument("--seed", type=int, default=1337, help="random seed")  
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")  
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

train_data_path = args.root_path
epoch_step = args.epoch_step

snapshot_path = "../model/" + args.exp + "/"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  
batch_size = args.batch_size * len(args.gpu.split(","))  
max_iterations = args.max_iterations  
base_lr = args.base_lr  
labeled_bs = args.labeled_bs  
model_num = args.model_num  
model_step = epoch_step // model_num  
model_is_first_term = [True] * model_num
ema_decay = args.ema_decay

max_score = 0
teacher_max_score = 0

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2 
patch_size = (256, 256)  
T = 0.1 
hyp = 0.01
Good_student = 0  

class RandomRotFlip_consistency_2D(object):
    """
    Randomly rotate and flip the image and label in a sample.
    Args:
        None
    """
    def __call__(self, image):
        k = random.randint(0, 3)  # Randomly choose rotation count
        # dims_ = random.choice([(2, 3), (2, 4), (3, 4)])
        dims_ = (2, 3)
        image = torch.rot90(image, k, dims=dims_)  # Rotate image
        axis = random.randint(2, 3)  # Randomly choose flip axis
        # print(dims_, axis)
        image = torch.flip(image, dims=(axis,))  # Flip image
        return image.clone(), k, dims_, axis 
    
randomRotFlip = RandomRotFlip_consistency_2D()

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
            out_main += net(torch.tensor(slice, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(1)) # 难怪是这里写得不正确，服了
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

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_teacher_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 10.0 * ramps.sigmoid_rampup(epoch, 20.0)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  # 版本不对应
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def rand_bbox(size, lam=None):
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cut_mix(image=None, mask=None, gt=None): # x
    mix_volume = image.clone()
    mix_target = mask.clone()
    if gt != None:
        mix_gt = gt.clone()
    u_rand_index = torch.randperm(image.size()[0])[:image.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(image.size(), lam=np.random.beta(4, 4))
    for i in range(0, args.batch_size):
        if i < mix_volume.shape[0]:
            mix_volume[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            if gt != None:
                mix_gt[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    mix_gt[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    if gt != None:
        return mix_volume, mix_target, mix_gt
    else:
        return mix_volume, mix_target

class IntervalSwitch:
    def __init__(self, total_range: int, num_intervals_min: int, num_intervals_max: int):
        self.T = total_range
        self.num_intervals_max = num_intervals_max
        self.num_intervals_min = num_intervals_min
        self.split_points = []
        current = 0
        while True:
            step = random.randint(self.num_intervals_min, self.num_intervals_max) 
            current += step
            if current >= self.T:
                break
            self.split_points.append(current)

    def __call__(self, counter: int) -> bool:
        for i, point in enumerate(self.split_points):
            if counter < point:
                return i % 2 == 0
        return len(self.split_points) % 2 == 0

class ModelData:
    def __init__(self, outputs, label, labeled_bs, x_range, y_range, is_first_term=False):
        self.x_range = x_range
        self.y_range = y_range
        self.dice_loss = DiceLoss(4)

        self.outputs = outputs
        self.label = label
        self.loss_seg = F.cross_entropy(self.outputs[:labeled_bs], label[:labeled_bs])
        self.outputs_soft = F.softmax(self.outputs, dim=1)
        self.loss_seg_dice = self.dice_loss(self.outputs_soft[:labeled_bs, :, :, :], label[:labeled_bs])
        self.predict = torch.max(self.outputs_soft[:labeled_bs, :, :, :], dim=1)[1]  
        self.mse_dist = consistency_criterion(self.outputs_soft[:labeled_bs, :, :, :], self._one_hot_encoder(label[:labeled_bs])).mean(1) 
        self.loss = torch.tensor(0.0).cuda()  
        self.is_first_term = is_first_term

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(4):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float() 

    def get_supervised_loss(self, diff_mask=None):
        if diff_mask is None:
            self.supervised_loss = self.loss_seg + self.loss_seg_dice
            self.loss = self.loss + self.supervised_loss
        else:
            self.mse = torch.sum(diff_mask * self.mse_dist) / (torch.sum(diff_mask) + 1e-16)  
            self.supervised_loss = (self.loss_seg + self.loss_seg_dice) + 0.5 * self.mse
            self.loss = self.loss + self.supervised_loss

    def add_teacher_loss(self, model_outputs, teacher_outputs, consistency_weight=0):
        teacher_outputs_soft = F.softmax(teacher_outputs, dim=1)
        model_outputs_soft = F.softmax(model_outputs, dim=1)

        teacher_outputs_clone = (teacher_outputs_soft.clone().detach())
        teacher_outputs_clone1 = torch.pow(teacher_outputs_clone, 1 / T)
        teacher_outputs_clone2 = torch.sum(teacher_outputs_clone1, dim=1, keepdim=True)
        teacher_outputs_PLable = torch.div(teacher_outputs_clone1, teacher_outputs_clone2)  

        model_dist = consistency_criterion(model_outputs_soft, teacher_outputs_PLable)
        b, v, w, h = model_dist.shape 
        model_dist = torch.sum(model_dist) / (b * v * w * h)  
    
        self.loss = self.loss + consistency_weight * model_dist

    def get_loss(self, Plabel, consistency_weight=0):
        consistency_dist = F.kl_div(torch.log_softmax(self.outputs.clamp(min=1e-6, max=1.0), dim=1), torch.softmax(Plabel.clamp(min=1e-6, max=1.0), dim=1).detach(), reduction='none').mean()
        self.loss = self.loss + consistency_weight * consistency_dist * hyp

    def get_weakstrong_loss(self, strong_outputs, consistency_weight=0):
        consistency_weakstrong_dist = F.kl_div(torch.log_softmax(self.outputs.clamp(min=1e-6, max=1.0), dim=1), torch.softmax(strong_outputs.clamp(min=1e-6, max=1.0), dim=1).detach(), reduction='none').mean()
        self.loss = self.loss + consistency_weight * consistency_weakstrong_dist * hyp

    def get_rot_loss(self, rot_outputs1, rot_outputs2, x_range, y_range, consistency_weight=0):
        consistency_rot_dist = 0
        for i in range(rot_outputs1.shape[0]):  
            A = self.outputs[i][:, self.x_range[0][i]:self.x_range[1][i], self.y_range[0][i]:self.y_range[1][i]]
            B = rot_outputs1[i][:, x_range[0][i]:x_range[1][i], y_range[0][i]:y_range[1][i]]
            C = rot_outputs2[i][:, x_range[0][i]:x_range[1][i], y_range[0][i]:y_range[1][i]]
            
            with torch.no_grad(): 
                B_soft = torch.softmax(B.detach(), dim=0)
                _, B_label = torch.max(B.detach(), dim=0)  
                C_soft = torch.softmax(C.detach(), dim=0)  
                _, C_label = torch.max(C.detach(), dim=0)  

                entropy_1 = -torch.sum(B_soft  * torch.log(B_soft + 1e-16), dim=0)
                entropy_2 = -torch.sum(C_soft * torch.log(C_soft + 1e-16), dim=0)

                weights_1 = torch.exp(-entropy_1) / (torch.exp(-entropy_1) + torch.exp(-entropy_2))
                weights_2 = 1 - weights_1

                # conflit_label = (B_label != C_label)
                consistency_label = (B_label == C_label)
            
            weighted_outputs = weights_1.unsqueeze(0) * B + weights_2.unsqueeze(0) * C 
            
            A = A.clamp(min=1e-6, max=1.0)
            weighted_outputs = weighted_outputs.clamp(min=1e-6, max=1.0) 

            if torch.sum(consistency_label) > 0:
                consistency_loss = (F.kl_div(torch.log_softmax(A, dim=0), torch.softmax(weighted_outputs, dim=0).detach(), reduction='none').sum(0) * consistency_label).sum() / torch.sum(consistency_label)
            else:
                consistency_loss = torch.tensor(0.0).cuda()

            consistency_rot_dist += consistency_loss / num_classes  

            del A, B, C, consistency_loss

        self.loss = self.loss + consistency_weight * consistency_rot_dist * hyp

def train_model(model_array, teacher_model_array, optimizer_array, data_buffer, model_iter_num, idx, first_term=False,):
    for sampled_batch in data_buffer:
        if first_term:     
            model_output_weak = model_array[idx](sampled_batch[idx]["image"].cuda())
            model_output_strong = model_array[idx](sampled_batch[idx]["image_strong"].cuda())
            teacher_model_output_weak = teacher_model_array[idx](sampled_batch[idx]["image"].cuda())
            teacher_model_output_strong = teacher_model_array[idx](sampled_batch[idx]["image_strong"].cuda())

            data_weak = ModelData(model_output_weak, sampled_batch[idx]["label"].cuda(), labeled_bs,
                            sampled_batch[idx]['x_range'], sampled_batch[idx]['y_range'], first_term,)
            teacher_data_weak = ModelData(teacher_model_output_weak, sampled_batch[idx]["label"].cuda(), labeled_bs,
                            sampled_batch[idx]['x_range'], sampled_batch[idx]['y_range'], first_term,)
            data_strong = ModelData(model_output_strong, sampled_batch[idx]["label"].cuda(), labeled_bs,
                            sampled_batch[idx]['x_range'], sampled_batch[idx]['y_range'], first_term,)
            teacher_data_strong = ModelData(teacher_model_output_strong, sampled_batch[idx]["label"].cuda(), labeled_bs,
                            sampled_batch[idx]['x_range'], sampled_batch[idx]['y_range'], first_term,)

            consistency_weight = get_current_consistency_weight(model_iter_num[idx] // 150)
            teacher_weight = get_teacher_consistency_weight(model_iter_num[idx] // 150)
            
            data_weak.get_supervised_loss()
            teacher_data_weak.get_supervised_loss()

            data_weak.get_weakstrong_loss(model_output_strong, consistency_weight)
            teacher_data_weak.get_weakstrong_loss(teacher_model_output_strong, consistency_weight)

            data_strong.add_teacher_loss(model_output_strong[labeled_bs:], teacher_model_output_weak[labeled_bs:], teacher_weight,)
            teacher_data_strong.add_teacher_loss(teacher_model_output_strong[labeled_bs:], model_output_weak[labeled_bs:], teacher_weight,)
            
            optimizer_array[idx].zero_grad()
            loss = data_weak.loss + teacher_data_weak.loss + data_strong.loss + teacher_data_strong.loss
            loss.backward()
            optimizer_array[idx].step()
            
            # update_ema_variables(model_array[idx], teacher_model_array[idx], ema_decay, model_iter_num[idx])

            model_iter_num[idx] += 1
        else:
            data_arrays_weak = []
            data_arrays_strong = []
            model_output_array_weak = []
            model_output_array_strong = []
            teacher_data_arrays_weak = []
            teacher_data_arrays_strong = []
            teacher_output_array_weak = []
            teacher_output_array_strong = []
            for i in range(model_num):  
                if i == idx:
                    model_output_array_weak.append(model_array[i](sampled_batch[i]["image"].cuda()))
                    model_output_array_strong.append(model_array[i](sampled_batch[i]["image_strong"].cuda()))
                    data_arrays_weak.append(ModelData(model_output_array_weak[i], sampled_batch[i]["label"].cuda(), labeled_bs, sampled_batch[i]['x_range'], 
                                        sampled_batch[i]['y_range']))
                    data_arrays_strong.append(ModelData(model_output_array_strong[i], sampled_batch[i]["label"].cuda(), labeled_bs, sampled_batch[i]['x_range'],
                                        sampled_batch[i]['y_range']))
                    teacher_output_array_weak.append(teacher_model_array[i](sampled_batch[i]["image"].cuda()))
                    teacher_data_arrays_weak.append(ModelData(teacher_output_array_weak[i], sampled_batch[i]["label"].cuda(), labeled_bs, sampled_batch[i]['x_range'],
                                        sampled_batch[i]['y_range']))
                    teacher_output_array_strong.append(teacher_model_array[i](sampled_batch[i]["image_strong"].cuda()))
                    teacher_data_arrays_strong.append(ModelData(teacher_output_array_strong[i], sampled_batch[i]["label"].cuda(), labeled_bs, sampled_batch[i]['x_range'],
                                        sampled_batch[i]['y_range']))
                else:
                    with torch.no_grad():
                        model_output_array_weak.append(None)
                        data_arrays_weak.append(None)
                        model_output_weak = model_array[i](sampled_batch[1-i]["image"].cuda())
                        data_weak = ModelData(model_output_weak, sampled_batch[1-i]["label"].cuda(), labeled_bs, sampled_batch[1-i]['x_range'], 
                                            sampled_batch[1-i]['y_range'])
                        model_output_array_strong.append(None)
                        data_arrays_strong.append(None)
                        teacher_output_array_weak.append(None)
                        teacher_data_arrays_weak.append(None)
                        teacher_output_weak = teacher_model_array[i](sampled_batch[1-i]["image"].cuda())
                        teacher_data_weak = ModelData(teacher_output_weak, sampled_batch[1-i]["label"].cuda(), labeled_bs, sampled_batch[1-i]['x_range'], 
                                            sampled_batch[1-i]['y_range'])  
                        teacher_output_array_strong.append(None)
                        teacher_data_arrays_strong.append(None)

            min_seg_model_arrays = [data_arrays_weak[idx], data_weak, teacher_data_weak]  
            if min_seg_model_arrays[0].loss_seg_dice <= min_seg_model_arrays[1].loss_seg_dice and min_seg_model_arrays[0].loss_seg_dice <= min_seg_model_arrays[2].loss_seg_dice:
                Good_student = 0
            elif min_seg_model_arrays[0].loss_seg_dice > min_seg_model_arrays[1].loss_seg_dice and min_seg_model_arrays[0].loss_seg_dice <= min_seg_model_arrays[2].loss_seg_dice:
                Good_student = 1
            elif min_seg_model_arrays[0].loss_seg_dice <= min_seg_model_arrays[1].loss_seg_dice and min_seg_model_arrays[0].loss_seg_dice > min_seg_model_arrays[2].loss_seg_dice:
                Good_student = 2
            else:
                Good_student = 3

            min_seg_teacher_model_arrays = [teacher_data_arrays_weak[idx], teacher_data_weak, data_weak]  
            if min_seg_teacher_model_arrays[0].loss_seg_dice <= min_seg_teacher_model_arrays[1].loss_seg_dice and min_seg_teacher_model_arrays[0].loss_seg_dice <= min_seg_teacher_model_arrays[2].loss_seg_dice:
                Teacher_Good_student = 0
            elif min_seg_teacher_model_arrays[0].loss_seg_dice > min_seg_teacher_model_arrays[1].loss_seg_dice and min_seg_teacher_model_arrays[0].loss_seg_dice <= min_seg_teacher_model_arrays[2].loss_seg_dice:
                Teacher_Good_student = 1
            elif min_seg_teacher_model_arrays[0].loss_seg_dice <= min_seg_teacher_model_arrays[1].loss_seg_dice and min_seg_teacher_model_arrays[0].loss_seg_dice > min_seg_teacher_model_arrays[2].loss_seg_dice:
                Teacher_Good_student = 2
            else:
                Teacher_Good_student = 3

            diff_mask = None
            teacher_diff_mask = None
            if Good_student == 1:             
                diff_mask = ((min_seg_model_arrays[0].label[:labeled_bs]) != (min_seg_model_arrays[1].predict))
            elif Good_student == 2:
                diff_mask = ((min_seg_model_arrays[0].label[:labeled_bs]) != (min_seg_model_arrays[2].predict))
            elif Good_student == 3:
                diff_mask1 = ((min_seg_model_arrays[0].label[:labeled_bs]) != (min_seg_model_arrays[1].predict))
                diff_mask2 = ((min_seg_model_arrays[0].label[:labeled_bs]) != (min_seg_model_arrays[2].predict))
                diff_mask = torch.logical_or(diff_mask1, diff_mask2)
                del diff_mask1, diff_mask2

            if  Teacher_Good_student == 1:             
                teacher_diff_mask = ((min_seg_teacher_model_arrays[0].label[:labeled_bs]) != (min_seg_teacher_model_arrays[1].predict))
            elif Teacher_Good_student == 2:
                teacher_diff_mask = ((min_seg_teacher_model_arrays[0].label[:labeled_bs]) != (min_seg_teacher_model_arrays[2].predict))
            elif Teacher_Good_student == 3:
                teacher_diff_mask1 = ((min_seg_teacher_model_arrays[0].label[:labeled_bs]) != (min_seg_teacher_model_arrays[1].predict))
                teacher_diff_mask2 = ((min_seg_teacher_model_arrays[0].label[:labeled_bs]) != (min_seg_teacher_model_arrays[2].predict))
                teacher_diff_mask = torch.logical_or(teacher_diff_mask1, teacher_diff_mask2)
                del teacher_diff_mask1, teacher_diff_mask2
            
            if switch(model_iter_num[idx]) and diff_mask != None:
                data_arrays_weak[idx].get_supervised_loss(diff_mask)
                del diff_mask
            else:
                data_arrays_weak[idx].get_supervised_loss()

            if teacher_switch(model_iter_num[idx]) and teacher_diff_mask != None:
                teacher_data_arrays_weak[idx].get_supervised_loss(teacher_diff_mask)
                del teacher_diff_mask
            else:
                teacher_data_arrays_weak[idx].get_supervised_loss()
        
            # Plabel = data_arrays_weak[1-idx].outputs 
            Plabel = data_weak.outputs
            
            # teacher_Plabel = teacher_data_arrays_weak[1-idx].outputs 
            teacher_Plabel = teacher_data_weak.outputs

            consistency_weight = get_current_consistency_weight(model_iter_num[idx] // 150)
            teacher_weight = get_teacher_consistency_weight(model_iter_num[idx] // 150)
            
            if switch(model_iter_num[idx]): 
            # if False:
                data_arrays_weak[idx].get_loss(Plabel, consistency_weight) 
            else:
                # pass
                sampled_image_rot1, k, dims_, axis = randomRotFlip(sampled_batch[1-idx]['image'].cuda())
                model_output_weak_rot1 = model_array[idx](sampled_image_rot1)
                model_output_weak_rot1 = torch.flip(model_output_weak_rot1, dims=(axis,))
                model_output_weak_rot1 = torch.rot90(model_output_weak_rot1, 4-k, dims=dims_)    

                sampled_image_rot2, k, dims_, axis = randomRotFlip(sampled_batch[1-idx]['image'].cuda())
                model_output_weak_rot2 = model_array[idx](sampled_image_rot2)
                model_output_weak_rot2 = torch.flip(model_output_weak_rot2, dims=(axis,))
                model_output_weak_rot2 = torch.rot90(model_output_weak_rot2, 4-k, dims=dims_)    

                data_arrays_weak[idx].get_rot_loss(model_output_weak_rot1, model_output_weak_rot2, sampled_batch[1-idx]['x_range'], sampled_batch[1-idx]['y_range'], consistency_weight)

                del sampled_image_rot1, model_output_weak_rot1, sampled_image_rot2, model_output_weak_rot2

            if teacher_switch(model_iter_num[idx]):  
            # if False:
                teacher_data_arrays_weak[idx].get_loss(teacher_Plabel, consistency_weight) 
            else:   
                # pass
                sampled_image_rot1, k, dims_, axis = randomRotFlip(sampled_batch[1-idx]['image'].cuda())
                teacher_output_weak_rot1 = teacher_model_array[idx](sampled_image_rot1)
                teacher_output_weak_rot1 = torch.flip(teacher_output_weak_rot1, dims=(axis,))
                teacher_output_weak_rot1 = torch.rot90(teacher_output_weak_rot1, 4-k, dims=dims_)

                sampled_image_rot2, k, dims_, axis = randomRotFlip(sampled_batch[1-idx]['image'].cuda())
                teacher_output_weak_rot2 = teacher_model_array[idx](sampled_image_rot2)
                teacher_output_weak_rot2 = torch.flip(teacher_output_weak_rot2, dims=(axis,))
                teacher_output_weak_rot2 = torch.rot90(teacher_output_weak_rot2, 4-k, dims=dims_)
                
                teacher_data_arrays_weak[idx].get_rot_loss(teacher_output_weak_rot1, teacher_output_weak_rot2, sampled_batch[1-idx]['x_range'], sampled_batch[1-idx]['y_range'], consistency_weight)
                del sampled_image_rot1, teacher_output_weak_rot1, sampled_image_rot2, teacher_output_weak_rot2

            data_arrays_weak[idx].get_weakstrong_loss(model_output_array_strong[idx], consistency_weight)
            teacher_data_arrays_weak[idx].get_weakstrong_loss(teacher_output_array_strong[idx], consistency_weight)

            sampled_batch_mix, plabel_model_output_array_mix = cut_mix(sampled_batch[idx]['image_strong'][labeled_bs:].cuda(), model_output_array_weak[idx][labeled_bs:])
            teacher_output_array_mix = teacher_model_array[idx](sampled_batch_mix)
            teacher_data_arrays_strong[idx].add_teacher_loss(teacher_output_array_mix, plabel_model_output_array_mix, teacher_weight,)
            
            sampled_batch_mix, plabel_teacher_output_array_mix = cut_mix(sampled_batch[idx]['image_strong'][labeled_bs:].cuda(), teacher_output_array_weak[idx][labeled_bs:])
            model_output_array_mix = model_array[idx](sampled_batch_mix)
            data_arrays_strong[idx].add_teacher_loss(model_output_array_mix, plabel_teacher_output_array_mix, teacher_weight,)  
            del sampled_batch_mix, plabel_model_output_array_mix, plabel_teacher_output_array_mix, teacher_output_array_mix, model_output_array_mix

            optimizer_array[idx].zero_grad()
            loss = data_arrays_weak[idx].loss + teacher_data_arrays_weak[idx].loss + data_arrays_strong[idx].loss + teacher_data_arrays_strong[idx].loss
            loss.backward()
            optimizer_array[idx].step()
            
            # Teacher 迭代
            # update_ema_variables(model_array[idx], teacher_model_array[idx], ema_decay, model_iter_num[idx])
     
            model_iter_num[idx] += 1
    
    if epoch_num >= 100:
        global max_score
        global teacher_max_score
        for model in model_array:
            model.eval()
        for teacher_model in teacher_model_array:
            teacher_model.eval()

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
        
        if metric_record[0] > max_score:
            max_score = metric_record[0]
            for i, model in enumerate(model_array):
                save_mode_path_vnet = os.path.join(snapshot_path, "pmt_" + str(i) + "_iter_" + str(model_iter_num[i]) + '_' + str(round(max_score, 4)) + ".pth")
                torch.save(model.state_dict(), save_mode_path_vnet)
                logging.info("save model to {}".format(save_mode_path_vnet))

        dataloader = iter(val_set)
        tbar = range(len(val_set))
        tbar = tqdm(tbar, ncols=135)
        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        for batch_idx in tbar:
            x, y = next(dataloader)
            y_tilde = test_single_case(teacher_model_array, x)
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
        teacher_metric_record = (first_total + second_total + third_total) / (3 * len(val_set))
        print("teacher_metric_record: ", teacher_metric_record)
    
        if teacher_metric_record[0] > teacher_max_score:
            teacher_max_score = teacher_metric_record[0]
            for i, model in enumerate(teacher_model_array):
                save_mode_path_vnet = os.path.join(snapshot_path, "pmt_teacher_" + str(i) + "_iter_" + str(model_iter_num[i]) + '_' + str(round(teacher_max_score, 4)) + ".pth")
                torch.save(model.state_dict(), save_mode_path_vnet)
                logging.info("save teacher model to {}".format(save_mode_path_vnet))
        
        for model in model_array:
            model.train()
        for teacher_model in teacher_model_array:
            teacher_model.train()

if __name__ == "__main__":
    data_flag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_path = snapshot_path + '/' + data_flag
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")
    shutil.copytree(".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"]))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S",)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name):
        # Network definition
        if name == "unet":
            net  = UNet(in_chns=1, class_num=4)
            model = net.cuda()
        return model

    model_array = []
    teacher_model_array = []
    for i in range(model_num):
        model_array.append(create_model(name="unet"))
        teacher_model_array.append(create_model(name="unet"))

    db_train = ACDCDataset("/ldap_shared/home/***/PMT/data/ACDC",
                            "/ldap_shared/home/***/PMT/data/ACDC",
                            split="train")
    
    val_set = ACDCDataset("/ldap_shared/home/***/PMT/data/ACDC",
                              "/ldap_shared/home/***/PMT/data/ACDC",
                              split="test")

    labeled_idxs = list(range(256))  # todo set labeled num 136
    unlabeled_idxs = list(range(256, 1312))  # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn,)
    optimizer_array = []  
    for i in range(model_num):
        optimizer_array.append(optim.SGD(list(model_array[i].parameters()) + list(teacher_model_array[i].parameters()), lr=base_lr, momentum=0.9, weight_decay=0.0001,))
    lr_scheduler = PolyLR(start_lr=base_lr, lr_power=0.9, total_iters=max_iterations)

    # 这个东西效果很可能等于mse
    if args.consistency_type == "mse":
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == "kl":
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    
    max_epoch = max_iterations // len(trainloader) + 1
    # lr_ = base_lr
    for i in range(model_num):
        model_array[i].train()
        teacher_model_array[i].train()
    model_iterations = []
    for i in range(model_num):
        model_iterations.append(0)
    training_wl = []
    model_iter_num = [0] * model_num
    for i in range(model_num):
        training_wl.append(i)


    # warmup1 = 29.8 
    # warmup2 = 30

    # iter_num = 0
    # data_buffer = []
    # for i in tqdm(range(int(warmup1)), ncols=70):
    #     epoch_num = i
    #     for j in range(model_step // len(trainloader)):
    #         for i_batch, (normal_batch, cons_batch) in enumerate(trainloader):
    #             data_buffer.append((normal_batch, cons_batch))
    #             iter_num = iter_num + 1
    #     train_model(model_array, teacher_model_array, optimizer_array, data_buffer, model_iter_num, training_wl[0], model_is_first_term[training_wl[0]],)
    #     data_buffer = []
    #     current_lr = lr_scheduler.get_lr(cur_iter=iter_num)
    #     for _, opt_group in enumerate(optimizer_array[training_wl[0]].param_groups): 
    #         opt_group['lr'] = current_lr
    # data_buffer = []
    # fraction = warmup1 - int(warmup1)
    # if fraction > 0:
    #     epoch_num = int(warmup1) 
    #     for j in range(int(model_step // len(trainloader) * fraction)):
    #         for i_batch, (normal_batch, cons_batch) in enumerate(trainloader):
    #             data_buffer.append((normal_batch, cons_batch))
    #             iter_num = iter_num + 1
    #     train_model(model_array, teacher_model_array, optimizer_array, data_buffer, model_iter_num, training_wl[0], model_is_first_term[training_wl[0]],)
    #     current_lr = lr_scheduler.get_lr(cur_iter=iter_num)
    #     for _, opt_group in enumerate(optimizer_array[training_wl[0]].param_groups): 
    #         opt_group['lr'] = current_lr
    # model_is_first_term[training_wl[0]] = False
    # temp_idx = training_wl.pop(0)
    # training_wl.append(temp_idx)  

    # iter_num = 0
    # data_buffer = []
    # for i in tqdm(range(warmup2), ncols=70):
    #     epoch_num = i
    #     for j in range(model_step // len(trainloader)):
    #         for i_batch, (normal_batch, cons_batch) in enumerate(trainloader):
    #             data_buffer.append((normal_batch, cons_batch))
    #             iter_num = iter_num + 1
    #     train_model(model_array, teacher_model_array, optimizer_array, data_buffer, model_iter_num, training_wl[0], model_is_first_term[training_wl[0]],)
    #     data_buffer = []
    #     current_lr = lr_scheduler.get_lr(cur_iter=iter_num)
    #     for _, opt_group in enumerate(optimizer_array[training_wl[0]].param_groups):
    #         opt_group['lr'] = current_lr
    # model_is_first_term[training_wl[0]] = False
    # temp_idx = training_wl.pop(0)
    # training_wl.append(temp_idx)  


    iter_num = 0
    data_buffer = []
    switch = IntervalSwitch(max_iterations, 250, 1500)
    teacher_switch = IntervalSwitch(max_iterations, 250, 1500)
    for epoch_num in tqdm(range(max_epoch * len(trainloader) // model_step), ncols=70):  # max_epoch = max_iterations // len(trainloader) + 1
        for i in range(model_step // len(trainloader)):
            for i_batch, (normal_batch, cons_batch) in enumerate(trainloader):
                data_buffer.append((normal_batch, cons_batch))
                if iter_num >= max_iterations:
                    break
                iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        
        if iter_num >= max_iterations:  
            training_wl.pop(0)
            break
        
        train_model(model_array, teacher_model_array, optimizer_array, data_buffer, model_iter_num, training_wl[0], model_is_first_term[training_wl[0]],)

        current_lr = lr_scheduler.get_lr(cur_iter=iter_num)
        for _, opt_group in enumerate(optimizer_array[training_wl[0]].param_groups):
            opt_group['lr'] = current_lr
        
        model_is_first_term[training_wl[0]] = False
        temp_idx = training_wl.pop(0)
        training_wl.append(temp_idx)  
            
        if len(data_buffer) >= epoch_step: 
            for i in range(model_step): 
                data_buffer.pop(0)

    for i in training_wl:
        train_model(model_array, teacher_model_array, optimizer_array, data_buffer, model_iter_num, i, model_is_first_term[i],)
