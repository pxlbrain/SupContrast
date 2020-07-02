from __future__ import division
import os
import json
import glob

from torchvision import transforms
import torch.utils.data as data
from PIL import Image


# def build_transforms():
#
#     normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
#
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(size=160, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         normalize,
#     ])
#
#     return train_transform


class Ali(data.Dataset):
    def __init__(self,transform):
        self.train_json_path = "/media/deep-storage-2/AliProducts/train.json"
        self.train_dir_path = "/media/deep-storage-2/AliProducts/dataset/train/"
        self.train_info = self.get_imgs_info(self.train_json_path)

        self.classes = set()
        for e in self.train_info.keys():
            self.classes.add(self.train_info[e]["class_id"])
        self.classes = sorted(list(self.classes))
        self.clsid2label = {v: i for i, v in enumerate(self.classes)}

        self.train_img_paths = glob.glob(self.train_dir_path + "**/*.jpg", recursive=True)
        train_img_ids = [os.path.basename(path) for path in self.train_img_paths]
        self.train_labels = [
            self.clsid2label[self.train_info[img_id]["class_id"]] for img_id in train_img_ids
        ]

        self.paths = self.train_img_paths
        self.labels = self.train_labels
        self.totensor = transform

    @staticmethod
    def get_imgs_info(json_path):
        with open(json_path) as json_file:
            data_info = json.load(json_file)
        imgs_info = {}
        for img_info in data_info["images"]:
            img_id = img_info["image_id"]
            imgs_info[img_id] = img_info
        return imgs_info

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = self.paths[item]
        img = self.pil_loader(img_path)
        img = self.totensor(img)
        label = self.labels[item]
        return img, label

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')