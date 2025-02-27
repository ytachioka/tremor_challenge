#!/usr/bin/env python
# coding: utf-8

# predict labels files for final submission

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from data_loader import sliding_window
from config import rootdir
from prep import run_fixed_part, div_dataset, load_data
from models.HAR_model import Transformer

def load_model_for_test(exp_name,model_type):
    model = model_type()
    model.load_state_dict(torch.load(os.path.join(rootdir,exp_name,'HAR_model.pth'), weights_only=True))
    model = model.to(device)
    model.eval()

    return model


def test_model(exp_names, test_data_dict, model_type):

    models = []
    for exp_name in exp_names:
        models.append(load_model_for_test(exp_name,model_type))

    eval_step = 5

    with open(os.path.join(rootdir,'label.pkl'),'rb') as f:
        label_dict = pickle.load(f)
    label_dict_r = {}
    for k,v in label_dict.items():
        label_dict_r[v] = k

    df_all = None

    with torch.no_grad():
        for u, df in test_data_dict.items():
            print('Testing data from user %s.'% u)

            df['Subject'] = u

            # Extract sensor data and labels
            seq_ids = np.unique(df['seq'].values)

            for seq in seq_ids:
                df_q = df[df['seq']==seq]
                raw_data = df_q[selected_columns[:3]].values
                label = np.reshape(df_q[selected_columns[-1]].values,(-1,1))

                data = np.concatenate((raw_data,label),axis=1)
                data_b = sliding_window(data,len_sw=300,step=eval_step)

                sample = data_b[:,:,0:3].astype(np.float32)
                sample = torch.tensor(sample, device=device)

                y_prob = None
                for model in models:
                    output = model(sample)
                    y_prob1 = torch.softmax(output,dim=1).detach().cpu().numpy()
                    if y_prob is None:
                        y_prob = y_prob1
                    else:
                        y_prob += y_prob1
                
                # Get the predicted class labels
                y_pred = np.argsort(-y_prob.sum(axis=0))
                y_pred2 = np.argmax(y_prob, axis=1)  # output Shape: (batch_size,)

                df['label'] = df['label'].mask(df['seq']==seq,str([int(label_dict_r[i]) for i in np.argsort(y_pred).tolist()]))
                #df['label'] = df['label'].mask(df['seq']==seq,np.bincount(y_pred2).argmax())
            
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all,df])

    df_all.to_csv(os.path.join(rootdir,exp_names[0],'test.csv'))


def run_baseline1(exp_name):

    test_model([exp_name], test_data_dict, model_type=Transformer)

def run_baseline1_multimodels(exp_names):

    test_model(exp_names, test_data_dict, model_type=Transformer)


if __name__ == '__main__':

    seed_value, expdir = 2025, 'exp'
    #seed_value, expdir = 40, 'exp2'
    selected_columns, new_columns = run_fixed_part(seed_value)

    user_paths, train_users, val_users, test_users = div_dataset()
    train_data_dict, val_data_dict, test_data_dict = load_data(user_paths, train_users, val_users, test_users, selected_columns)

    #run_baseline1(os.path.join(expdir,'HAR','virtual_opt_cvae_2model'))

    exp_names = ['exp','exp2','exp3']
    exp_names = [os.path.join(e,'HAR','virtual_opt_cvae_2model') for e in exp_names]

    run_baseline1_multimodels(exp_names)

