import os
import numpy as np
import tensorflow as tf

from tflearn.data_utils import image_preloader  # shuffle,
from statistics import mean,stdev
from load_data.load_dataset import DatasetLoader
from DNN.net import *   # Network architectures
import h5py
import tflearn

class TRAIN_EVAL():
    def __init__(self, network, environment='GPU'):
        '''

        :param environment: GPU or CPU
        '''
        pass

    def load_data(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    pass

class TRAIN_EVAL_TFLEARN(TRAIN_EVAL):

    def train(self):
        pass

    def evaluate(self):
        pass
    pass


class TRAIN_EVAL_KERAS(TRAIN_EVAL):

    def train(self):
        pass

    def evaluate(self):
        pass

    pass