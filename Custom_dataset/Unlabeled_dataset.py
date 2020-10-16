import torchvision
import PIL as p

DatasetType = {
    'cifar10' : torchvision.datasets.CIFAR10,
    'cifar100' : torchvision.datasets.CIFAR100
    }

class Unlabeled_dataset_cifar(torchvision.datasets.CIFAR10):
    '''
    Custom dataset designed to return one element from the labeled dataset and mu elements from the unlabeled
    '''
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, item):
        datasample = p.Image.fromarray(self.data[item])
        return self.transform(datasample)


