import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb

from src.Data_processing.Paper_dataset import *


class Trainer:

    def __init__(self, dataset, loss_function, use_gpu=True, data_to_train=0.5, data_to_test=0.25, data_to_eval=0.25):
        '''
        :param data_path: path to the data folder
        :param use_gpu: true if the program should use GPU
        :param data_to_train: percent of data to train
        :param data_to_test: percent of data to test
        :param data_to_eval: percent of data to eval
        '''
        self.dataset = dataset
        self.data_to_train = data_to_train
        self.data_to_test = data_to_test
        self.data_to_eval = data_to_eval
        self.loss_function = loss_function
        self.main_device = self.get_main_device(use_gpu)

    def get_main_device(self, use_gpu):
        '''
           Initilize class by loading data and maybe preprocess
           ASSIGN CUDAS IF POSSIBLE
           :return:
        '''

        # Check if gpu is available
        if torch.cuda.is_available() and use_gpu:
            device = "cuda:0"
            print("Using GPU")
            self.check_gpu_card()
        else:
            device = "cpu"
            print("Using CPU")

        # assign gpu or cpu to the main_device
        main_device = torch.device(device)

        return main_device

    def check_gpu_card(self):
        '''
        The method tries to check which gpu you are using on your computer
        :return:
        '''
        try:
            import pycuda.driver as cudas
            print("The device you are using is: ", cudas.Device(0).name())

        except ImportError as e:
            print("Could not find pycuda and thus not show amazing stats about youre GPU, have you installed CUDA?")
            pass

    def train(self, model, batch_size, learn_rate, epochs):

        '''
        Performs a train and test cycle at the given model
        :param model: a trainable pytorch model
        :param batch_size: wanted size of each batch
        :param learn_rate: wanted learning rate
        :param learn_decay: wanted learn decay
        :param learn_momentum: wanted learn momentum
        :param epochs: number of epochs
        :return:
        '''

