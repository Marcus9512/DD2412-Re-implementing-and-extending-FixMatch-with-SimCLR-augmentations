import torch
from torch.utils.data import Dataset

class L_U_Dataset(Dataset):
    '''
    Custom dataset designed to return one element from the labeled dataset and mu elements from the unlabeled
    '''
    def __init__(self, labeled, unlabeled, mu):
        self.len = len(labeled)
        self.mu = mu
        self.unlabeled = unlabeled
        self.labeled = labeled

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        index_pos_2 = item * self.mu
        labeled_data = self.labeled[item]
        unlabeled_data = torch.cat(tuple(self.unlabeled[i][0] for i in range(index_pos_2, index_pos_2 + self.mu)))

        return labeled_data, unlabeled_data
