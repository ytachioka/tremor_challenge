#!/usr/bin/env python
# coding: utf-8

# # Virtual Data Generation for Tremor challenge

import os
import numpy as np
import pandas as pd

from config import rootdir
from prep import run_fixed_part, div_dataset, load_data, get_folder_size


def save_virtual_data(data, filename):
    '''
    Participants can use this function to save csv data to /data/virtual/
    :param data: dataframe type, shape is (data length, dim=7), columns = new_columns = selected_columns[:6] + [selected_columns[-1]]
    :return:
    '''

    data.to_csv(os.path.join(virt_directory, filename+'.csv'), index=False)
    return


import torch
class sample_model:
    def __init__(self,exp_name, model_type):
        #print(raw_data.shape)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model_type()
        model.load_state_dict(torch.load(os.path.join(rootdir,exp_name,'AE_model.pth'), weights_only=True))
        model.to(self.device)
        self.model = model

    def sample(self,n_data):
        with torch.no_grad():
            self.model.eval()
            if os.path.isfile(os.path.join(virt_directory,'ratio.npy')):
                labels = np.zeros(n_data,dtype=int)
                ratio = np.load(os.path.join(virt_directory,'ratio.npy'))
                r = np.random.rand(n_data)
                for i in range(n_data):
                    for j in range(len(ratio)-1):
                        if ratio[j] <= r[i] and r[i] < ratio[j+1]:
                            labels[i] = j
                            break
                labels = torch.tensor(labels).to(self.device)
            else:
                labels = torch.tensor(np.arange(0,n_data)%11).to(self.device)
            kwargs = {'device':self.device, 'labels':labels}
            output = self.model.sample(n_data,**kwargs)

        return output.detach().cpu().numpy(), labels.detach().cpu().numpy()


def flatten_data(output,labels):
    import pickle
    with open(os.path.join(rootdir,'label.pkl'),'rb') as f:
        label_dict = pickle.load(f)
    label_dict_r = {}
    for k,v in label_dict.items():
        label_dict_r[v] = k
    
    # Concatenate data with operation labels
    #virtual_data = np.concatenate([output, labels], axis=1)
    n_b, n_s = output.shape[0], output.shape[1]
    virtual_data = np.zeros((n_b*n_s,4),dtype=float)
    for i in range(n_b):
        for j in range(n_s):
            virtual_data[i*n_s+j,0:3] = output[i,j,:]
        virtual_data[(i*n_s):((i+1)*n_s),3] = label_dict_r[labels[i]]

    return virtual_data


def custom_virtual_data_generation(train_data_dict):
    '''
    This function aims to generate virtual and from train_data_dict, and save the data to virt_directory.
    Participants could not change the input and output of this function.
    Participants could modify the code inside this function.
    During the code submission, participants need to submit this function and its relavant functions, such as custom_virtual_data_generation_algorithm.
    '''
    import torch
    import pickle
    from data_loader import generate_dataloader, vote_labels, load_data_label, sliding_window

    exp_name, model_type, num_models = get_ids()
    model = sample_model('exp/'+exp_name, model_type)

    if num_models > 1:
        models = [model]
        for i in range(2,num_models+1):
            models.append(sample_model(f'exp{i}/'+exp_name, model_type))
    
    with open(os.path.join(rootdir,'label.pkl'),'rb') as f:
        label_dict = pickle.load(f)

    if os.path.isfile(os.path.join(virt_directory,'params.npy')):
        sampling_ratio, step = np.load(os.path.join(virt_directory,'params.npy'))
        step = int(step)
    else:
        sampling_ratio, step = 0.3, 200
    
    print(f'sampling_ratio {sampling_ratio}, step {step}')

    for u, df in train_data_dict.items():
        print('Generating virtual data from user %s.'% u)

        # Extract sensor data and labels
        raw_data = df[selected_columns[:3]].values
        label = np.reshape(df[selected_columns[-1]].apply(lambda x: label_dict[x]).values,(-1,1))
        data = np.concatenate((raw_data,label),axis=1)
        data_b = sliding_window(data,len_sw=300,step=step)

        batch_size = len(data_b)
        batch_size_s = max(int(sampling_ratio*batch_size),1)
        batch_size_r = batch_size - batch_size_s
        idx = np.random.choice(batch_size,batch_size_r)

        sample = data_b[idx,:,0:3]
        label = data_b[idx,:,-1]

        sample = torch.tensor(sample, device=model.device, dtype=torch.float)
        label = torch.tensor(label, device=model.device, dtype=torch.long) # (batchsize,lensw)
        vote_label = vote_labels(label) #(batchsize,1)
        vote_label = vote_label.to(model.device)
        kwargs = {'labels':vote_label}
     
        if num_models > 1:
            virtual_data = []
            vote_label_np = vote_label.detach().cpu().numpy()
            for i in range(0,len(models)):
                ## sampling
                output_s, labels_s = models[i].sample(n_data=batch_size_s)
                virtual_data.append(flatten_data(output_s,labels_s))

                ## reconstruction
                __, output_r = models[i].model(sample,**kwargs)
                output_r = output_r.detach().cpu().numpy()
                virtual_data.append(flatten_data(output_r,vote_label_np))

            virtual_data = np.concatenate(virtual_data,axis=0)
        else:
            output, labels = model.sample(n_data=1000)
            virtual_data = flatten_data(output,labels)

        # Convert np.array to dataframe
        df = pd.DataFrame(virtual_data, columns=new_columns)

        # Example of virtual data structure
        df.head(3)

        # Save data to /data/virtual/
        save_virtual_data(df, u)
        # df.to_csv(os.path.join(virt_directory, u+'.csv'), index=False)

def get_ids():
    from models.AE_model import VAE, CVAE
    #exp_name, model_type = 'pretrain/vae', VAE
    exp_name, model_type = 'pretrain/cvae', CVAE

    num_models = 2

    return exp_name, model_type, num_models


if __name__ == '__main__':

    selected_columns, new_columns = run_fixed_part()

    user_paths, train_users, val_users, test_users = div_dataset()
    train_data_dict, val_data_dict, test_data_dict = load_data(user_paths, train_users, val_users, test_users, selected_columns)

    exp_name, model_type, num_models = get_ids()

    v_dir = 'virtual_opt_'
    #v_dir = 'virtual_'
    
    if num_models > 1:
        v_dir += exp_name.split(os.sep)[-1] + f'_{num_models}model'
    else:
        v_dir += exp_name.split(os.sep)[-1]

    virt_directory = os.path.join(rootdir,'data',v_dir)
    os.makedirs(virt_directory,exist_ok=True)

    custom_virtual_data_generation(train_data_dict)

    # Example usage
    size = get_folder_size(virt_directory)
    print(f"The total size of the folder is: {size} bytes ({size/1024/1024} MB)")


