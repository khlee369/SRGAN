import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as T
import cv2


class FaceData(Dataset):
    def __init__(self):
        cwd = os.getcwd()

        self.input_data_path = os.path.join(cwd, "./trainData/input")
        self.gt_data_path = os.path.join(cwd, "./trainData/gt")
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