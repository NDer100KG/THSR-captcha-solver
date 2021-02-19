# -*- coding: utf-8 -*-
"""Main module for training

This module is use for training the CNN model to solving 
Taiwan High Speed Rail booking system captcha.

Example:

        $ python train.py
"""
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
from tensorboardX import SummaryWriter
from tqdm import tqdm


def loss_(scores, labels):
    """Custom loss function.

    Args:
        scores (tensor): ground truth.
        labels (tensor): CNN output.

    Returns:
        tensor: The loss value.
    """
    digit1_cross_entropy = torch.nn.functional.cross_entropy(scores[0], labels[:, 0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(scores[1], labels[:, 1])
    digit3_cross_entropy = torch.nn.functional.cross_entropy(scores[2], labels[:, 2])
    digit4_cross_entropy = torch.nn.functional.cross_entropy(scores[3], labels[:, 3])
    return digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy


def train(path=None, log_path=None):
    """Train the CNN mode.

    Args:
        path (str): checkpoint file path.
        log_path (str): log_path. default='./log/train_<datetime>.log'

    """

    """ ===== Constant var. start ====="""
    train_comment = ""
    pre_process = False
    use_gpu = True
    num_workers = 7
    batch_size = 128
    lr = 0.001
    lr_decay = 0.9
    max_epoch = 500
    stat_freq = 10
    model_name = "test"
    """ ===== Constant var. end ====="""

    # step0: init. log and checkpoint dir.
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoints_dir = "./checkpoints/" + model_name
    if len(train_comment) > 0:
        checkpoints_dir = checkpoints_dir + "_" + train_comment
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    if log_path == None:
        if not os.path.isdir("./log"):
            os.mkdir("./log")
        if not os.path.exists('./log/' + model_name):
            os.makedirs("./log/" + model_name)
        log_path = "./log/{}".format(model_name)

    # step1: dataset
    val_data = Data(train=False, pre_process=pre_process)
    val_dataloader = DataLoader(val_data, 100, num_workers=num_workers)

    train_data = Data(train=True, pre_process=pre_process)
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    writer = SummaryWriter(log_path)

    with open(log_path + "/log.txt", "w") as log_file:

        # step2: instance and load model
        model = CNN()

        if path != None:
            print('using mode "{}"'.format(path))
            print('using mode "{}"'.format(path), file=log_file, flush=True)
            model.load(path)
        else:
            print("init model by orthogonal_", file=log_file, flush=True)
            for name, param in model.named_parameters():
                if len(param.shape) > 1:
                    torch.nn.init.orthogonal_(param)

        if use_gpu:
            model.cuda()

        # summary(model, (1, 128, 128))

        # step3: loss function and optimizer
        criterion = loss_
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

        previous_loss = 1e100
        # epoch loop
        for epoch in range(max_epoch):
            running_loss = 0.0
            total_loss = []

            # batch loop
            pbar = tqdm(enumerate(train_dataloader))
            for i, (data, label) in pbar:
                input = data
                target = label
                if use_gpu:
                    input = input.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                score = model(input)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_loss.append(loss.item())
                if (i + 1) % stat_freq == 0:
                    pbar.set_description("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / stat_freq))
                    writer.add_scalar('train/loss', running_loss / stat_freq, epoch* len(train_dataloader) + i)
                    running_loss = 0.0

            previous_loss = np.mean(total_loss)
            acc_img, acc_digit = val(model, val_dataloader, use_gpu)
            writer.add_scalar('eval/acc_img', acc_img, epoch * len(train_dataloader))
            writer.add_scalar('eval/acc_digit', acc_digit, epoch * len(train_dataloader))
            save_path = "{}/{}_{}.pth".format(
                checkpoints_dir, time_str, epoch + 1, acc_img, acc_digit
            )
            model.save(save_path)
            print(
                '{} epoch : {}, acc_img : {}, acc_digit : {}, loss : {}, path : "{}"'.format(
                    datetime.now(), epoch, acc_img, acc_digit, previous_loss, save_path
                ),
                file=log_file,
                flush=True,
            )
            print(
                "acc_img : {}, acc_digit : {}, loss : {}".format(acc_img, acc_digit, previous_loss)
            )
            print('path : "{}"'.format(save_path))
            if np.mean(total_loss) > previous_loss:
                lr = lr * lr_decay
                print("reduce loss from to {}".format(lr))


def decode(scores):
    """Decode the CNN output.

    Args:
        scores (tensor): CNN output.

    Returns:
        list(int): list include each digit index.
    """
    tmp = np.array(tuple(map(lambda score: score.cpu().numpy(), scores)))
    tmp = np.swapaxes(tmp, 0, 1)
    return np.argmax(tmp, axis=2)


def val(model, dataloader, use_gpu):
    """val. the CNN model.

    Args:
        model (nn.model): CNN model.
        dataloader (dataloader): val. dataset.

    Returns:
        tuple(int, in): average of image acc. and digit acc..
    """
    model.eval()  # turn model to eval. mode(enable droupout layers...)
    result_digit = []
    result_img = []
    for i, (data, label) in enumerate(dataloader):
        with torch.no_grad():  # disable autograd
            if use_gpu:
                input = data.cuda()
            score = model(input)
            tmp = decode(score) == label.numpy()
            result_digit += tmp.tolist()
            result_img += np.all(tmp, axis=1).tolist()

    # turn model back to training mode.
    model.train()

    return np.mean(result_img), np.mean(result_digit)

if __name__ == "__main__":
    train()
