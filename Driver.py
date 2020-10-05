'''
Driver class for the network.
Our "main" method.
'''

import logging
import argparse
import torch

import Trainer


LOGGER_NAME = "Driver"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)
    args = parser.parse_args()
    logger.info(f"Dataset path: {args.base_path} ")


    #model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False, num_classes=1)
    #loss_function =
    #dataset =

    #trainer = Trainer(dataset, loss_function)
    #trainer.train()