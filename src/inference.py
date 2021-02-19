import torch
import numpy as np
import sys
import os
import dataset
from datetime import datetime
from model import CNN
from dataset import Data
from PIL import Image
from torch.utils.data import DataLoader
from torchsummary import summary

if __name__ == "__main__":
    test_data = Data(dir = 'dataset', train=False, pre_process=False)
    test_dataloader = DataLoader(test_data, 100, num_workers=8)

    for i, (data, label) in enumerate(test_dataloader):
        
        print()


'''
def test(img_path, model_path, use_gpu=False):
    """!!! Useless !!!
    """
    model = CNN()
    model.load(model_path)
    if use_gpu:
        model.cuda()
    char_table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # 把模型设为验证模式
    from torchvision import transforms as T

    transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(),])

    data = dataset.transforms(img_path, pre=False).unsqueeze(dim=0)
    model.eval()
    with torch.no_grad():
        if use_gpu:
            data = data.cuda()
        score = model(data)
        score = decode(score)
        score = "".join(map(lambda i: char_table[i], score[0]))
        return score
        '''
