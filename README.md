# Recreation of FixMatch
This repository is a recreation of the FixMatch algorithm by Sohn, et al in PyTorch. The paper introducing the FixMatch
altorithm can be found [here](https://arxiv.org/abs/2001.07685). The FixMatch algorithm is semi-supervised learning algorithm
that exploit unlabeled samples to increase the performance. The model can be trained with two datasets, CIFAR-10 and CIFAR-100.

## Install requirements
Run the following command in the terminal to install requirements.
`pip install -r requirements.txt`
If you want more information about the available GPU install pycuda by:
`pip install pycuda`
*Note* pycuda requiers [CUDA](https://developer.nvidia.com/cuda-downloads) and [Visual Studio](https://visualstudio.microsoft.com/)

## Run the network
* Run `python Driver.py`
* Arguments, dataset is the only required argument `--dataset, --mu, --batch_size, --epochs, --num_labels, --checkpoint_ratio and --resume`
* `--dataset` Decides the dataset, can be set to CIFAR10 or CIFAR100
* `--mu` Mu is a scaling factor for the unlabeled batch size. The unlabeled batch size will be mu*(labeled batch size). Default value: 7
* `--batch_size` The size of a labeled batch. Default value: 64
* `--epochs` Number of epochs. Default value: 200
* `--num_labels` The amount of labeled data for each class in the dataset. Default value: 400
* `--checkpoint_ratio` How often a backup of the network will be saved. Default value: 50 (epochs)
* `--resume` To start the network from a checkpoint, assign the path, for example, resume=Saved_checkpoints/name_of_checkpoint.pt.tar.    

## Tensorboard
* You need tensorflow installed `pip install tensorflow`
* Start tensorboard with `tensorboard --logdir=runs`
