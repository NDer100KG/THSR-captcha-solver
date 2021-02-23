# -*- coding: utf-8 -*-
"""Dataset module for training

This module include Dataset class and some utils for training.

"""
import torch
import os
import random
import numpy as np
from torchvision import transforms as T
from torch.utils import data
from pre_process import pre_process
from PIL import Image
import pandas as pd


def table(c):
    """Tranfer char to index.

    Args:
        c (str): one char.

    Returns:
        int: index of c.
    """
    i = ord(c)
    if i < 65:
        i -= 48
    else:
        i -= 55
    return i


class Data(data.Dataset):
    def __init__(self, train, dir="dataset/train", train_val_ratio=0.9):
        self.dir = dir
        self.load_csv()
        self.pre_process = pre_process
        self.train_val_ratio = train_val_ratio
        if train:
            self.data = self.data[: int(len(self.data) * self.train_val_ratio)]
        else:
            self.data = self.data[int(len(self.data) * self.train_val_ratio) :]

    def load_csv(self, name="labels.csv"):
        data = pd.read_csv(os.path.join(self.dir, name), skiprows=None)
        data = data.values
        self.data = data.tolist()

    def __getitem__(self, index):
        name, label = self.data[index]
        c_label = np.array(list(map(table, label)))

        data = self.transforms(name)
        return data, torch.tensor(c_label, dtype=torch.long), label

    def transforms(self, img_name):
        data = Image.open(os.path.join(self.dir, img_name)).convert("L")
        f_transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(),])
        data = f_transforms(data)
        return data

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import cv2, sys

    t = Data(train=True)

    for i in range(len(t)):
        img, c_label, label = t[i]

        print(label)
        cv2.imshow("tmp", img.numpy()[0, :, :, np.newaxis])
        k = cv2.waitKey()
        if k == 27:
            sys.exit()
