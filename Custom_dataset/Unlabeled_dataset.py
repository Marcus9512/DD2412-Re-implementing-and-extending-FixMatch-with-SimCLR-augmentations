import torchvision
import PIL as p

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
    '''
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, item):
        datasample = p.Image.fromarray(self.data[item])
        return self.transform(datasample)

class Unlabeled_dataset_cifar10(torchvision.datasets.CIFAR10):
    '''
    Custom dataset designed to return a 2 transformations of the loaded image for CIFAR-10
    '''
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, item):
        datasample = p.Image.fromarray(self.data[item])
        return self.transform(datasample)


