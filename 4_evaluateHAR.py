#!/usr/bin/env python
# coding: utf-8

# # HAR model recognition for Tremor challenge
# 
import os
import numpy as np
import torch
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from data_loader import load_data_label, sliding_window
from config import rootdir
from prep import run_fixed_part, div_dataset, load_data
from models.HAR_model import Transformer

def test_model(exp_name, test_data_dict, model_type, exp_name2 = None):
    model = model_type()
    model.load_state_dict(torch.load(os.path.join(rootdir,exp_name,'HAR_model.pth'), weights_only=True))
    model = model.to(device)
    model.eval()
    if exp_name2 is not None:
        model2 = model_type()
        model2.load_state_dict(torch.load(os.path.join(rootdir,exp_name2,'HAR_model.pth'), weights_only=True))
        model2 = model2.to(device)
        model2.eval()

    fout = open(os.path.join(rootdir,exp_name,'test.log'),'wt')
    def fprint(string):
        print(string)
        fout.write(string+'\n')
        fout.flush()
    
    eval_step = 5

    with open(os.path.join(rootdir,'label.pkl'),'rb') as f:
        label_dict = pickle.load(f)
    label_dict_r = {}
    for k,v in label_dict.items():
        label_dict_r[v] = k

    with torch.no_grad():
        for u, df in test_data_dict.items():
            print('Testing data from user %s.'% u)

            # Extract sensor data and labels
            seq_ids = np.unique(df['seq'].values)

            for seq in seq_ids:
                df_q = df[df['seq']==seq]
                raw_data = df_q[selected_columns[:3]].values
                label = np.reshape(df_q[selected_columns[-1]].values,(-1,1))

                data = np.concatenate((raw_data,label),axis=1)
                data_b = sliding_window(data,len_sw=300,step=eval_step)

                sample = data_b[:,:,0:3]
                sample = torch.tensor(sample, device=device, dtype=torch.float)

                output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
                y_prob = torch.softmax(output,dim=1).detach().cpu().numpy()
                if exp_name2 is not None:
                    output2 = model2(sample)
                    y_prob2 = torch.softmax(output2,dim=1).detach().cpu().numpy()
                    y_prob += y_prob2
                    

                # Get the predicted class labels
                y_pred = np.argsort(-y_prob.sum(axis=0))
                y_pred2 = np.argmax(y_prob, axis=1)  # output Shape: (batch_size,)

                df['label'] = df['label'].mask(df['seq']==seq,str([int(label_dict_r[i]) for i in np.argsort(y_pred).tolist()]))

                #df['label'] = df['label'].mask(df['seq']==seq,np.bincount(y_pred2).argmax())
            df.to_csv(os.path.join(rootdir,exp_name,f'{u}.csv'))


def run_baseline1(exp_name):

    test_model(exp_name, test_data_dict, model_type=Transformer)

def run_baseline1_2model(exp_name1,exp_name2):

    test_model(exp_name1, test_data_dict, model_type=Transformer, exp_name2=exp_name2)

def run_baseline2(virt='virtual'):
    ## just add virtual data
    exp_name = expdir+'/HAR/'+virt
    add_data=os.path.join('data',exp_name.split(os.sep)[-1])

    __, __, test_data = load_data_label(train_data_dict,val_data_dict,test_data_dict,new_columns,add_data=add_data)
    test_model(exp_name, test_data, test_data_dict, model_type=Transformer)


if __name__ == '__main__':

    seed_value, expdir = 2025, 'exp'
    #seed_value, expdir = 40, 'exp2'
    selected_columns, new_columns = run_fixed_part(seed_value)

    user_paths, train_users, val_users, test_users = div_dataset()
    train_data_dict, val_data_dict, test_data_dict = load_data(user_paths, train_users, val_users, test_users, selected_columns)

    #run_baseline1(exp_name=os.path.join(expdir,'HAR','virtual_opt_cvae_2model'))
    run_baseline1_2model(exp_name1=os.path.join(expdir,'HAR','virtual_opt_cvae_2model'),exp_name2=os.path.join('exp2','HAR','virtual_opt_cvae_2model'))

