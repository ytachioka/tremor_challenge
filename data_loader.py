import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
from config import rootdir
import pickle

# ## 3.2 データローダーの準備
class data_loader_OpenPack(Dataset):
    def __init__(self, samples, labels, device='cpu'):
        self.samples = torch.tensor(samples).to(device)  # check data type
        self.labels = torch.tensor(labels)  # check data type

    def __getitem__(self, index):
        target = self.labels[index]
        sample = self.samples[index]
        return sample, target

    def __len__(self):
        return len(self.labels)

def sliding_window(datanp, len_sw, step):
    '''
    :param datanp: shape=(data length, dim) raw sensor data and the labels. The last column is the label column.
    :param len_sw: length of the segmented sensor data
    :param step: overlapping length of the segmented data
    :return: shape=(N, len_sw, dim) batch of sensor data segment.
    '''

    # generate batch of data by overlapping the training set
    data_batch = []
    for idx in range(0, datanp.shape[0] - len_sw - step, step):
        data_batch.append(datanp[idx: idx + len_sw, :])
    data_batch.append(datanp[-1 - len_sw: -1, :])  # last batch
    xlist = np.stack(data_batch, axis=0)  # [B, data length, dim]

    return xlist

def generate_dataloader(data, device, batch_size = 512, len_sw = 300, step = 150, if_shuffle=True):
    tmp_b = sliding_window(data, len_sw, step)
    data_b = tmp_b[:, :, :-1]
    label_b = tmp_b[:, :, -1]
    data_set_r = data_loader_OpenPack(data_b, label_b, device=device)
    data_loader = DataLoader(data_set_r, batch_size=batch_size,
                              shuffle=if_shuffle, drop_last=False)
    return data_loader

def vote_labels(label):
    # Iterate over each sample in the batch
    votel = []
    for i in range(label.size(0)):
        # Get unique labels and their counts
        unique_labels, counts = label[i].unique(return_counts=True)

        # Find the index of the maximum count
        max_count_index = counts.argmax()

        # Get the label corresponding to that maximum count
        mode_label = unique_labels[max_count_index]

        # Append the mode to the result list
        votel.append(mode_label)

    # Convert the result list to a tensor and reshape to (batch, 1)
    vote_label = torch.tensor(votel, dtype=torch.long).view(-1)
    return vote_label


def load_data_label(train_data_dict,val_data_dict,test_data_dict,new_columns,add_data=None):
    # ## 3.1 実フォルダと仮想フォルダの両方からデータとラベルを読み取る
    # 

    # real and virtual training data

    ## real data
    train_data = []
    for u, data in train_data_dict.items():
        train_data.append(data[new_columns].values)
        # print(data[new_columns].values.shape)

    if add_data is not None:
        ## virtual data
        # find csv files in 'data/virtual'
        virt_paths = []
        for root, dirs, files in os.walk(add_data):
            for file in files:
                if file.endswith('.csv'):
                    virt_paths.append(os.path.join(root, file))
        print(f'Virtual csv file paths are as shown follows: {virt_paths}')
        assert len(virt_paths) > 0

        for p in virt_paths:
            # Load the CSV file with only the selected columns
            data = pd.read_csv(p, usecols=new_columns)
            train_data.append(data.values)

    train_data = np.concatenate(train_data, axis=0)
    print('Shape of train data is %s'%str(train_data.shape))

    # validatation and test data
    val_data = []
    for u, data in val_data_dict.items():
        val_data.append(data[new_columns].values)

    test_data = []
    for u, data in test_data_dict.items():
        test_data.append(data[new_columns].values)

    val_data = np.concatenate(val_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)

    print('Shape of validation data is %s'%str(val_data.shape))
    print('Shape of test data is %s'%str(test_data.shape))

    # convert operation ID to labels (from 0 to n)
    if os.path.isfile(os.path.join(rootdir,'label.pkl')):
        with open(os.path.join(rootdir,'label.pkl'),'rb') as f:
            label_dict = pickle.load(f)
    else:
        labels = np.unique(train_data[:, -1])
        label_dict = dict(zip(labels, np.arange(len(labels))))
        with open(os.path.join(rootdir,'label.pkl'),'wb') as fout:
            pickle.dump(label_dict,fout)
    train_data[:,-1] = np.array([label_dict[i] for i in train_data[:,-1]])
    val_data[:,-1] =  np.array([label_dict[i] for i in val_data[:,-1]])
    test_data[:,-1] =  -1 #np.array([label_dict[i] for i in test_data[:,-1]])

    return train_data, val_data, test_data
