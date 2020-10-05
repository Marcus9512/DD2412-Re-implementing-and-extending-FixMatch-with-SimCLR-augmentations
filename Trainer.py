import os
import torch
import logging
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb

from os import path
from datetime import datetime

LOGGER_NAME = "Trainer"


class Trainer:

    def __init__(self, dataset, loss_function, batch_size=10, use_gpu=True, workers=2):
        '''
        :param data_path: path to the data folder
        :param use_gpu: true if the program should use GPU
        :param data_to_train: percent of data to train
        :param data_to_test: percent of data to test
        :param data_to_eval: percent of data to eval
        '''

        # setup logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(LOGGER_NAME)

        self.logger = logger
        self.dataset = dataset
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.workers = workers

        # setup GPU if possible
        self.main_device = self.get_main_device(use_gpu)

        # setup of tensrobord summary writer, is used to print graphs
        self.summary = tb.SummaryWriter()

    def get_main_device(self, use_gpu):
        '''
           Initilize class by loading data and maybe preprocess
           ASSIGN CUDAS IF POSSIBLE
           :return:
        '''

        # Check if gpu is available
        if torch.cuda.is_available() and use_gpu:
            device = "cuda:0"
            self.logger.info("Using GPU")
            self.check_gpu_card()
        else:
            device = "cpu"
            self.logger.info("Using CPU")

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
            self.logger.info("The device you are using is: {cudas.Device(0).name()}")

        except ImportError as e:
            self.logger.warn(
                "Could not find pycuda and thus not show amazing stats about youre GPU, have you installed CUDA?")
            pass

    def split_train_dataset(self, percent_to_validation):
        assert percent_to_validation < 1.0
        assert percent_to_validation > 0

        size_of_dataset = len(self.dataset["train_set"])
        train, val = ut.random_split(self.dataset["train_set"], [int((1.0 - percent_to_validation)*size_of_dataset),
                                                                 int(percent_to_validation*size_of_dataset)])
        return train, val

    def validate_directory(self, save_path):
        '''
        Makes sure that the directory under save_path exists, otherwise creates it.
        :param save_path:
        :return:
        '''
        if not path.exists(save_path):
            self.logger.warn("The directory Saved_networks was not found, creating the directory")
            try:
                os.mkdir(save_path)
            except OSError as exc:
                raise

    def save_network(self, model):
        '''
        Saves the model into a directory named Saved_networks
        :param model:
        :return:
        '''
        # create a path to saved networks
        save_path = os.path.dirname("Data")
        save_path = os.path.join(save_path, "Saved_networks")

        # make sure that the directory exists
        self.validate_directory(save_path)

        # create a timestamp
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%d-%m-%Y_%H-%M-%S")

        final_path = save_path + "/" + timestamp
        # save network
        torch.save(model.state_dict(), final_path)

        return final_path

    def close_summary(self):
        self.summary.flush()
        self.summary.close()

    def train(self, model, learn_rate, weight_decay, momentum, epochs=10, percent_to_validation=0.2):
        '''

        :param model:
        :param batch_size:
        :param learn_rate:
        :param weight_decay:
        :param momentum:
        :param epochs:
        :param percent_to_validation:
        :return: a path of the saved model
        '''
        # set model to GPU or CPU
        model.to(self.main_device)

        train, val = self.split_train_dataset(percent_to_validation)

        # Create dataloders for each part of the dataset
        train_dataloader = ut.DataLoader(train, batch_size=self.batch_size, shuffle=True,
                                         num_workers=self.workers, pin_memory=True)
        val_dataloader = ut.DataLoader(val, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.workers, pin_memory=True)

        # select optimizer type, current is SGD
        optimizer = opt.SGD(model.parameters(), lr=learn_rate, weight_decay=weight_decay, momentum=momentum)

        # set the wanted loss function to criterion
        criterion = self.loss_function

        for e in range(epochs):
            self.logger.info(f"Epoch {e} of {epochs}")

            for session in ["training", "validation"]:
                if session == "training":
                    current_dataloder = train_dataloader
                    model.train()
                else:
                    current_dataloder = val_dataloader
                    model.eval()

                combined_loss = 0
                i = 0
                for data in current_dataloder:
                    sample, label = data

                    # Send sample and label to GPU or CPU
                    sample = sample.to(device=self.main_device)
                    label = label.to(device=self.main_device)

                    if session == "training":
                        # Reset gradients between training
                        optimizer.zero_grad()
                        out = model(sample)
                    else:
                        # Disable gradient modifications
                        with torch.no_grad():
                            out = model(sample)

                    # Calculate loss
                    loss = criterion(out, label)
                    combined_loss += loss.item()

                    if session == "training":
                        # Backprop
                        loss.backward()
                        optimizer.step()

                    if (i % 1000 == 0):
                        self.logger.info(f"{session} img: {i}")
                    i += 1
                combined_loss /= len(current_dataloder)
                self.logger.info(f"{session} loss: {combined_loss}")
                self.summary.add_scalar('Loss/' + session, combined_loss, e)

        return self.save_network(model)

    def test(self, save_path, model):
        test_dataloader = ut.DataLoader(self.dataset["test_set"], batch_size=self.batch_size, shuffle=True,
                                        num_workers=self.workers, pin_memory=True)

        model.load_state_dict(torch.load(save_path))

        number_of_testdata = 0
        correct = 0
        for data in test_dataloader:
            sample, label = data

            # Send sample and label to GPU or CPU
            sample = sample.to(device=self.main_device, dtype=torch.float32)
            label = label.to(device=self.main_device, dtype=torch.float32)

            with torch.no_grad():
                out = model(sample)
                _, pred = torch.max(out, 1)

                for v in (pred == label):
                    if v:
                        correct += v.sum().item()
                number_of_testdata += label.size(0)

        self.logger.info(f"Accuracy: {(correct / number_of_testdata) * 100}")
