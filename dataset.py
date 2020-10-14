import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as T
import cv2

train_path = "./../CelebA-HQ_Dataset/Train/"
val_path = "./../CelebA-HQ_Dataset/Val/"

input_folder = "HQ_32x32"
gt_folder = "HQ_128x128"

class FaceData(Dataset):
    def __init__(self, dset):
        cwd = os.getcwd()
        if dset == 'train':
            dset_path = train_path
        else:
            dset_path = val_path

        self.input_data_path = os.path.join(cwd, dset_path + input_folder)
        self.gt_data_path = os.path.join(cwd, dset_path + gt_folder)
        self.input_data_list = os.walk(self.input_data_path).__next__()[2]
        self.gt_data_list = os.walk(self.gt_data_path).__next__()[2]
        self.input = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.gt = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def __getitem__(self, idx):
        input_img = Image.open(os.path.join(self.input_data_path, self.input_data_list[idx])).convert("RGB")
        gt_img = Image.open(os.path.join(self.gt_data_path, self.gt_data_list[idx])).convert("RGB")
        return self.input(input_img), self.gt(gt_img)


    def __len__(self):
        return len(self.gt_data_list)