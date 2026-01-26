import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case_array

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="/ldap_shared/home/***/PMT/data/LA",
    help="Name of Experiment",
)  # todo change dataset path
parser.add_argument(
    "--model", type=str, default="PMT", help="model_name"
)  # todo change test model name
parser.add_argument("--gpu", type=str, default="3", help="GPU to use")
parser.add_argument("--model_num", type=int, default=2, help="num of models")
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
snapshot_path = "../model/" + FLAGS.model + "/"
test_save_path = "../model/prediction/" + FLAGS.model + "_post/"
model_num = FLAGS.model_num
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + "/test.list", "r") as f:  # todo change test flod
    image_list = f.readlines()
image_list = [
    FLAGS.root_path + "/data/" + item.replace("\n", "") + "/mri_norm2.h5" for item in image_list
]


def create_model(name="vnet"):
    # Network definition
    if name == "vnet":
        net = VNet(
            n_channels=1,
            n_classes=num_classes,
            normalization="batchnorm",
            has_dropout=True,
        )
        model = net.cuda()
    return model


def test_calculate_metric(epoch_num):
    model_array = []
    model_save_path1 = ''
    model1 = create_model(name="vnet")
    
    model1.eval()
    model1.load_state_dict(torch.load(model_save_path1))
    model_array.append(model1)   

    avg_metric = test_all_case_array(
        model_array,
        image_list,
        num_classes=num_classes,
        patch_size=(112, 112, 80),
        stride_xy=18,
        stride_z=4,
        save_result=True,
        test_save_path=test_save_path,
    )

    return avg_metric


if __name__ == "__main__":
    iters = 24000
    metric = test_calculate_metric(iters)
    # print("iter:", iter)
    print(metric)
