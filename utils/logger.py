import random
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import torch
from torchvision import datasets, transforms
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')


# ---------------------- Logger ----------------------
class Logger(logging.Logger):
    '''Inherit from logging.Logger.
    Print logs to console and file.
    Add functions to draw the training log curve.'''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        super(Logger, self).__init__('Training logger')

        # Print logs to console and file
        file_handler = logging.FileHandler(
            os.path.join(self.dir_path, 'train_log.txt'))
        console_handler = logging.StreamHandler()
        log_format = logging.Formatter(
            "%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        self.addHandler(file_handler)
        self.addHandler(console_handler)

        # Draw curve
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="Training loss")
        self.ax1 = self.fig.add_subplot(122, title="Testing CMC/mAP")
        self.x_epoch_loss = []
        self.x_epoch_test = []
        self.y_train_loss = []
        self.y_test = {}
        self.y_test['top1'] = []
        self.y_test['mAP'] = []

    def save_curve(self):
        self.ax0.plot(self.x_epoch_loss, self.y_train_loss,
                      'bs-', markersize='2', label='test')
        self.ax0.set_ylabel('Training')
        self.ax0.set_xlabel('Epoch')
        self.ax0.legend()

        self.ax1.plot(self.x_epoch_test, self.y_test['top1'],
                      'rs-', markersize='2', label='top1')
        self.ax1.plot(self.x_epoch_test, self.y_test['mAP'],
                      'bs-', markersize='2', label='mAP')
        self.ax1.set_ylabel('%')
        self.ax1.set_xlabel('Epoch')
        self.ax1.legend()

        save_path = os.path.join(self.dir_path, 'train_log.jpg')
        self.fig.savefig(save_path)

    def save_img(self, fig):
        plt.imsave(os.path.join(self.dir_path, 'rank_list.jpg'), fig)
