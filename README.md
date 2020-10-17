# Recreation of FixMatch
## Install requirements
Run the following command in the terminal to install requirements.
`pip install -r requirements.txt`
If you want more information about the available GPU install pycuda by:
`pip install pycuda`
*Note* pycuda requiers [CUDA](https://developer.nvidia.com/cuda-downloads) and [Visual Studio](https://visualstudio.microsoft.com/)

## Run the network
* Run `python Driver.py`
* Arguments, dataset is the only required argument `--dataset, --mu, --batch_size, --epochs, --num_labels`
* Available datasets `CIFAR10 and CIFAR100`

## Tensorboard
* You need tensorflow installed `pip install tensorflow`
* Start tensorboard with `tensorboard --logdir=runs`
