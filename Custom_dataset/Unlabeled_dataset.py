import torchvision
import PIL as p
import torch.utils.data as ut
import numpy as np

DatasetType = {
    'cifar10' : torchvision.datasets.CIFAR10,
    'cifar100' : torchvision.datasets.CIFAR100
    }

'''
These classes acts as extension to CIFAR-10 and CIFAR-100 to enable both weak and strong augmentation of the data.
'''
class Unlabeled_dataset_cifar100(torchvision.datasets.CIFAR100):
    '''
    Custom dataset designed to return a 2 transformations of the loaded image for CIFAR-100

    Influenced by: https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR100
    '''
    def __init__(self, root, train, transform, data_indicies):
        super().__init__(root=root, train=train, download=False, transform=transform)
        self.data = self.data[data_indicies]
        self.targets = np.array(self.targets)[data_indicies]

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = p.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class Unlabeled_dataset_cifar10(torchvision.datasets.CIFAR10):
    '''
    Custom dataset designed to return a 2 transformations of the loaded image for CIFAR-10

    Influenced by: https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
    '''
    def __init__(self, root, train, download, transform, data_indicies):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.data = self.data[data_indicies]
        self.targets = np.array(self.targets)[data_indicies]

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = p.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class Unlabeled_subset(ut.Subset):

    def __init__(self, dataset, index, transform):
        super().__init__(dataset, index)
        self.transform = transform

    def __getitem__(self, item):
        (data, target) = self.dataset[self.indices[item]]
        t = self.transform(data)

        print(t)
        #print(type(target))
        return self.transform(data), target