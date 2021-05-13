# adapted from https://github.com/facebookresearch/EGG

import numpy as np
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, path):
        self.frame = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                vector, label = line.split(';')
                vector = [float(i) for i in vector.split()]

                sender_input = torch.tensor(vector)
                label = torch.tensor([int(label)])
                self.frame.append((sender_input, label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

