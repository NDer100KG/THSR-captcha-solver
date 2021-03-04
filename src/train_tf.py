# -*- coding: utf-8 -*-
import tensorflow as tf
import torch
import numpy as np
import sys
import os, cv2
import dataset
from datetime import datetime
from model import CNN_tf
from dataset import Data
from PIL import Image
from torch.utils.data import DataLoader
from torchsummary import summary
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np


class Loss(tf.keras.layers.Layer):
    def __init__(self):
        super(Loss, self).__init__()
        self.sce = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, scores, labels):
        sce0 = self.sce(labels[:, 0], scores[0])
        sce1 = self.sce(labels[:, 1], scores[1])
        sce2 = self.sce(labels[:, 2], scores[2])
        sce3 = self.sce(labels[:, 3], scores[3])

        return sce0 + sce1 + sce2 + sce3


def train(path=None, log_path=None):
    """Train the CNN mode.

    Args:
        path (str): checkpoint file path.
        log_path (str): log_path. default='./log/train_<datetime>.log'

    """

    """ ===== Constant var. start ====="""
    train_comment = ""
    num_workers = 7
    batch_size = 64
    lr = 0.001
    lr_decay = 0.9
    max_epoch = 500
    stat_freq = 10
    model_name = "0304_tf_2"
    """ ===== Constant var. end ====="""

    # step0: init. log and checkpoint dir.
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
        if not os.path.exists("./log/" + model_name):
            os.makedirs("./log/" + model_name)
        log_path = "./log/{}".format(model_name)

    # step1: dataset
    val_data = Data(train=False, format="NHWC")
    val_dataloader = DataLoader(val_data, 100, num_workers=num_workers)

    train_data = Data(train=True, format="NHWC")
    train_dataloader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )

    writer = tf.summary.create_file_writer(log_path)
    best_acc_img = 0

    # step2: instance and load model
    inputs = tf.keras.Input(shape=(128, 128, 1))
    model = CNN_tf()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=model)
    model.summary()

    # step3: loss function and optimizer
    criterion = Loss()
    optimizer = tf.keras.optimizers.Adam(1e-3)

    global_step = tf.Variable(1)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=global_step,)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoints_dir, max_to_keep=10)

    previous_loss = 1e100

    @tf.function
    def train_step(inputss, targets):
        with tf.GradientTape() as tape:
            pred = model(inputss, training=True)
            loss = criterion(pred, targets)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    # epoch loop
    for epoch in range(max_epoch):
        running_loss = 0.0
        total_loss = []

        # batch loop
        pbar = tqdm(enumerate(train_dataloader))
        for i, (data, label) in pbar:
            inputs = data.numpy()
            target = label.numpy()

            loss = train_step(inputs, target)

            running_loss += loss
            total_loss.append(loss)
            if (i + 1) % stat_freq == 0:
                pbar.set_description(
                    "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / stat_freq)
                )
                with writer.as_default():
                    tf.summary.scalar(
                        "train/loss",
                        running_loss / stat_freq,
                        step=epoch * len(train_dataloader) + i,
                    )

                running_loss = 0.0

        previous_loss = np.mean(total_loss)
        acc_img, acc_digit, im_show = val(model, val_dataloader)

        im_show[0] = cv2.putText(
            im_show[0], "".join(Data.decode(im_show[1])), (20, 20), 2, 1, (255, 0, 255)
        )
        with writer.as_default():
            tf.summary.scalar("eval/acc_img", acc_img, epoch * len(train_dataloader))
            tf.summary.scalar("eval/acc_digit", acc_digit, epoch * len(train_dataloader))
            tf.summary.image("img", im_show[0][np.newaxis, :, :, :], epoch * len(train_dataloader))

        if acc_img > best_acc_img:
            # ckpt_manager.save()
            model.save(checkpoints_dir, save_format = 'h5')
            print(
                "acc_img : {}, acc_digit : {}, loss : {}".format(acc_img, acc_digit, previous_loss)
            )
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


def val(model, dataloader):
    """val. the CNN model.

    Args:
        model (nn.model): CNN model.
        dataloader (dataloader): val. dataset.

    Returns:
        tuple(int, in): average of image acc. and digit acc..
    """
    result_digit = []
    result_img = []

    @tf.function
    def val_step(inputs):
        pred = model(inputs, training=False)
        return pred

    for i, (data, label) in enumerate(dataloader):
        inputs = data.numpy()
        target = label.numpy()

        score = val_step(inputs)
        pred = decode(score)

        tmp = pred == label.numpy()
        result_digit += tmp.tolist()
        result_img += np.all(tmp, axis=1).tolist()

    i = np.random.randint(0, len(dataloader) - 1)
    im_show = inputs[i]
    im_show = np.repeat((im_show * 255).astype(np.uint8), 3, -1)
    # turn model back to training mode.

    return np.mean(result_img), np.mean(result_digit), [im_show, pred[i]]


if __name__ == "__main__":
    train()
