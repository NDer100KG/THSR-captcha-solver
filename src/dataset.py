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
from PIL import Image
import pandas as pd

dic19 = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "7": 4,
    "9": 5,
    "a": 6,
    "c": 7,
    "f": 8,
    "h": 9,
    "k": 10,
    "m": 11,
    "n": 12,
    "p": 13,
    "q": 14,
    "r": 15,
    "t": 16,
    "y": 17,
    "z": 18,
}


class Data(data.Dataset):
    def __init__(self, train, dir=["dataset/train"], train_val_ratio=0.9):
        self.dir = dir
        self.load_csv()
        self.train_val_ratio = train_val_ratio
        if train:
            self.data = self.data[: int(len(self.data) * self.train_val_ratio)]
        else:
            self.data = self.data[int(len(self.data) * self.train_val_ratio) :]

    def load_csv(self, name="labels.csv"):
        self.data = []
        for d in self.dir:
            data = pd.read_csv(os.path.join(d, name), skiprows=None)
            data = data.values
            data = data.tolist()
            data = [[d + "/ori/" + _[0].replace("jpg", "png"), _[1]] for _ in data]
            self.data += data

    def __getitem__(self, index):
        name, label = self.data[index]
        code = self.encode(label)
        data = self.transforms(name)
        return data, torch.tensor(code, dtype=torch.long)

    def transforms(self, img_name):
        data = Image.open(img_name).convert("L")
        f_transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(),])
        data = f_transforms(data)
        return data

    def __len__(self):
        return len(self.data)

    def encode(self, label):
        return np.stack([dic19[l.lower()] for l in label])

    @staticmethod
    def decode(code):
        # if decode_dict is None:
        decode_dict = {value: key for key, value in dic19.items()}

        result = []
        for c in code:
            result.append(decode_dict[int(c)])
        return result


if __name__ == "__main__":
    import cv2, sys

    t = Data(train=True, dir = ["dataset/train"])

    for i in range(len(t)):
        img, code = t[i]  # (1, 128, 128)

        label = Data.decode(code)
        print(label)
        cv2.imshow("tmp", img.numpy()[0, :, :, np.newaxis])
        k = cv2.waitKey()
        if k == 27:
            sys.exit()
