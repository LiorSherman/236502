import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file,
                 max_num_files=None):

        data = np.load(file)

        if max_num_files is not None:
            max_num_files = min(len(data), max_num_files)
            self.data = data[np.random.choice(data.shape[0], max_num_files, replace=False), :]
        else:
            self.data = data

        print(f'Dataset shape : {self.data.shape}')

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()
