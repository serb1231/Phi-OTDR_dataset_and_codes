import numpy as np
import os
import scipy.io as scio
from torch.utils.data import Dataset


def normalize(data):                           # Min-max normalise to the range 0-255
    # Scales the whole (10000, 12) sample so its minimum becomes 0 and its maximum 255,
    # then rounds to integers (like a greyscale image).
    rawdata_max = data.max()
    rawdata_min = data.min()
    if rawdata_max == rawdata_min:             # flat signal: avoid division by zero
        return np.zeros_like(data)
    return np.round(255 * (data - rawdata_min) / (rawdata_max - rawdata_min)).astype(data.dtype)


class MyDataset(Dataset):
    # PyTorch dataset for the CNN. Each item is one .mat sample (10000 time points x
    # 12 spatial points) with its class label (0-5).

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir          # folder containing the class sub-folders
        self.names_file = names_file      # label.txt: "<relative path> <label>" per line
        self.transform = transform
        self.names_list = []
        if not os.path.isfile(self.names_file):
            raise FileNotFoundError(self.names_file + ' does not exist!')
        with open(self.names_file) as file:
            for line in file:
                parts = line.split()
                if len(parts) < 2:        # skip empty lines
                    continue
                data_path = self.root_dir + parts[0]
                if not os.path.isfile(data_path) or os.path.getsize(data_path) == 0:
                    print('Warning: skipping missing or empty file ' + data_path)
                    continue
                self.names_list.append((parts[0], int(parts[1])))
        self.size = len(self.names_list)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rel_path, label = self.names_list[idx]
        data_path = self.root_dir + rel_path
        rawdata = scio.loadmat(data_path)['data']  # (10000, 12) uint16
        rawdata = rawdata.astype(int)       # int64 so the arithmetic below cannot overflow
        data = normalize(rawdata)
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
