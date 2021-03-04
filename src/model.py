# -*- coding: utf-8 -*-
"""Model module

This module is the definition of CNN model.
"""
from tensorflow.python.training.tracking import base
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L


class base_block(tf.keras.layers.Layer):
    def __init__(
        self, channel, stride, trainable=True, name=None, dtype=None, dynamic=False, **kwargs
    ):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.conv1 = L.Conv2D(channel, 3, strides=stride, padding="same", use_bias=False)
        self.relu1 = L.ReLU()
        self.bn1 = L.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        return x


class CNN_tf(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.hidden1_1 = base_block(32, 2)
        self.hidden1_2 = base_block(32, 1)
        self.hidden2_1 = base_block(64, 2)
        self.hidden2_2 = base_block(64, 1)
        self.hidden3_1 = base_block(128, 2)
        self.hidden3_2 = base_block(128, 1)
        self.hidden4_1 = base_block(256, 2)
        self.hidden4_2 = base_block(256, 1)
        self.hidden5 = base_block(256, 2)

        self.Flatten = L.Flatten()
        self.dense1 = L.Dense(19, activation= tf.nn.softmax)
        self.dense2 = L.Dense(19, activation= tf.nn.softmax)
        self.dense3 = L.Dense(19, activation= tf.nn.softmax)
        self.dense4 = L.Dense(19, activation= tf.nn.softmax)

    def call(self, inputs, **kwargs):
        x = self.hidden1_1(inputs)
        x = self.hidden1_2(x)
        x = self.hidden2_1(x)
        x = self.hidden2_2(x)
        x = self.hidden3_1(x)
        x = self.hidden3_2(x)
        x = self.hidden4_1(x)
        x = self.hidden4_2(x)
        x = self.hidden5(x)

        x = self.Flatten(x)
        digit1 = self.dense1(x)
        digit2 = self.dense2(x)
        digit3 = self.dense3(x)
        digit4 = self.dense4(x)

        return digit1, digit2, digit3, digit4

    @staticmethod
    def decode(code):
        """Decode the CNN output.

        Args:
            scores (tensor): CNN output.

        Returns:
            list(int): list include each digit index.
        """
        tmp = np.array(tuple(map(lambda score: score.cpu().numpy(), code)))
        tmp = np.swapaxes(tmp, 0, 1)
        return np.argmax(tmp, axis=2)


class Flatten(nn.Module):
    """Flatten Module(layer).
    
       This model flatten input to (batch size, -1)
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN(nn.Module):
    """CNN modle.
    
       Refernce https://github.com/JasonLiTW/simple-railway-captcha-solver 
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.hidden3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.hidden4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.hidden5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, stride=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.flatten = Flatten()
        self.digit1 = nn.Linear(6400, 19)
        self.digit2 = nn.Linear(6400, 19)
        self.digit3 = nn.Linear(6400, 19)
        self.digit4 = nn.Linear(6400, 19)

    def forward(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)  # (2, 256, 5, 5)
        x = self.flatten(x)
        digit1 = torch.nn.functional.softmax(self.digit1(x), dim=1)
        digit2 = torch.nn.functional.softmax(self.digit2(x), dim=1)
        digit3 = torch.nn.functional.softmax(self.digit3(x), dim=1)
        digit4 = torch.nn.functional.softmax(self.digit4(x), dim=1)

        return digit1, digit2, digit3, digit4

    def save(self, path):
        """Save parameters of model.

        Args:
            path(str): parameters file path.

        """
        torch.save(self.state_dict(), path)
        # torch.save(self, path)

    def load(self, path):
        """Load parameters of model.

        Args:
            path(str): parameters file path.

        """
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        # torch.load(path)

    @staticmethod
    def decode(code):
        """Decode the CNN output.

        Args:
            scores (tensor): CNN output.

        Returns:
            list(int): list include each digit index.
        """
        tmp = np.array(tuple(map(lambda score: score.cpu().numpy(), code)))
        tmp = np.swapaxes(tmp, 0, 1)
        return np.argmax(tmp, axis=2)


if __name__ == "__main__":
    from torchsummary import summary

    ### torch model
    cnn = CNN()
    summary(cnn.cuda(), (1, 128, 128))

    ### tf model
    inputs = tf.keras.Input(shape=(128, 128, 1))
    cnn_tf = CNN_tf()(inputs)
    cnn_tf = tf.keras.Model(inputs=inputs, outputs=cnn_tf)
    cnn_tf.summary()
