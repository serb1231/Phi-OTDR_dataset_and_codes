import os
import scipy.io as scio
import numpy as np
from feature_extraction import feature_extraction


def get_diff_data(data):                       # Differentiate the data along time
    # data: (10000, 12) -> returns (9999, 12) where row i = data[i+1] - data[i].
    # Cast to a signed type first: the raw data is uint16, so a negative difference
    # would otherwise wrap around to a huge positive number (e.g. -1 -> 65535).
    return np.diff(data.astype(np.int64), axis=0)


def get_feature_list(data):                     # data: (10000, 12) or (9999, 12)
    # Extract the 16 features independently for each of the 12 spatial channels.
    n_channels = data.shape[1]
    sample_feature_list = np.zeros([n_channels, 16])    # (12, 16)
    for i in range(n_channels):
        f_data = data[:, i]                         # one channel: (10000,) or (9999,)
        sample_feature_list[i, :] = feature_extraction(f_data)   # 16 features
    return sample_feature_list


def read_label_file(rootpath, labelpath):
    # Each line of label.txt looks like "/01_background/xxx.mat 0".
    # Returns a list of (relative_path, label) tuples, skipping empty lines and
    # samples whose .mat file is missing or empty (0 bytes) with a warning.
    entries = []
    with open(labelpath) as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue
            path = rootpath + parts[0]
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                print('Warning: skipping missing or empty file ' + path)
                continue
            entries.append((parts[0], int(parts[1])))
    return entries


def get_das_data(rootpath, labelpath):
    # Build the SVM input matrix: one row of 12 * 32 = 384 features per sample.
    entries = read_label_file(rootpath, labelpath)
    temp = np.empty([len(entries), 12, 32])
    label_temp = np.empty(len(entries))
    for i, (rel_path, label) in enumerate(entries):
        path = rootpath + rel_path
        rawdata = scio.loadmat(path)['data']              # raw data (10000, 12), uint16
        diffdata = get_diff_data(rawdata)               # differentiated data (9999, 12)
        rawdata_sample_feature_list = get_feature_list(rawdata)   # raw-data features (12, 16)
        diffdata_sample_feature_list = get_feature_list(diffdata)  # diff-data features (12, 16)
        # Concatenate both feature sets per channel -> (12, 32)
        temp[i, :, :] = np.concatenate((rawdata_sample_feature_list, diffdata_sample_feature_list), axis=1)
        label_temp[i] = label
    temp = temp.reshape(len(entries), -1)  # (n_samples, 12*32) -- flattened
    return temp, label_temp
