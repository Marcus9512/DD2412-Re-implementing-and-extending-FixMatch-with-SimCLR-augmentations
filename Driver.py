'''
Driver class for the network.
Our "main" method.
'''

import logging
import argparse
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from Models.Wideresnet import *
from Custom_dataset.Unlabeled_dataset import *
from augmentation import  *

from Trainer import *


LOGGER_NAME = "Driver"

def get_normalization():
    '''
    Based on nomalisation example from:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    :return:
    '''
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#

def get_dataset(arg):
    '''
    Retruns a detaset in the formate  {"train_set", "test_set", "classes", "name"}
    :param arg:
    :return:
    '''

    def get_return_format(train, test, unlabeled, num_classes, name):
        return {
            "train_set": train,
            "unlabeled": unlabeled,
            "test_set": test,
            "num_classes": num_classes,
            "name": name
        }

    transform = get_normalization()

    if arg.lower() == "cifar10":
        # Based from pytorch Cifar10, https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        train = torchvision.datasets.CIFAR10(root='./Data', train=True, download=True, transform = transform)
        test = torchvision.datasets.CIFAR10(root='./Data', train=False, download=True, transform = transform)
        unlabeled = Unlabeled_dataset_cifar(root='./Unlabeled', train=True, download=True,
                                            transform= Wrapper(get_weak_transform(), get_strong_transform("CIFAR10")))

        return get_return_format(train, test, unlabeled, 10, "CIFAR10")

    elif arg.lower() == "cifar100":
        train = torchvision.datasets.CIFAR100(root='./Data', train=True, download=True, transform = transform)
        test = torchvision.datasets.CIFAR100(root='./Data', train=False, download=True, transform = transform)
        return get_return_format(train, test, 100, "CIFAR100")

    return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Which dataset should be used, supported: CIFAR10", required=True)
    parser.add_argument("--mu", type=int, help="Value for mu", default=7)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=1)

    args = parser.parse_args()
    logger.info(f"Selected dataset: {args.dataset}")

    dataset = get_dataset(args.dataset)

    if dataset == None:
        logger.error(f"Could not find dataset: {args.dataset}, terminating program")
        exit(1)

    #model = torch.hub.load('pytorch/vision:v0.6.0', 'wideresnet50_2', pretrained=False, num_classes=10)
    model = Wide_ResNet(28, 2, 0.3, 10)
    loss_function = nn.CrossEntropyLoss()

    trainer = Trainer(dataset, loss_function, batch_size=args.batch_size, mu=args.mu)
    path = trainer.train(model, learn_rate=0.03, weight_decay=0.0005, momentum=1e-9, epochs=args.epochs, num_labels=400, threshold=0.95)
    trainer.test(path, model)
    trainer.close_summary()
