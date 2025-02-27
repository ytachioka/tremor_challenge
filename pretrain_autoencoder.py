#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/qingxinxia/OpenPackChallenge2025/blob/main/1.Data%20Augmentation%20Algorithm%20with%20HAR.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Virtual Data Generation for Complex Industrial Activity Recognition
# 
# 
# 
# This notebook has been designed for the 7th Factory Work Activity Recognition Challenge competition with the aim of Activity Recognition using REAL Accelerometer from OpenPack dataset and GENERATED Accelerometer created by participants.
# 
# If you have any questions, please feel free to email qingxinxia@hkust-gz.edu.cn with the subject Factory Work Activity Recognition Challenge.
# 
# About this dataset and challenge -> https://abc-research.github.io/challenge2025/
# 
# This notebook was prepared by Qingxin Xia, Kei Tanigaki and Yoshimura Naoya.
# 
# ---
# 
# # 工場作業行動認識のための仮想データ生成
# 本ノートブックは、第7回工場作業行動認識チャレンジのために設計されました。本チャレンジでは、OpenPackデータセット作成に用いられた実際の加速度計と参加者によって生成された加速度センサデータを使用しています。
# 
# ご質問がある場合は、件名を「工場作業行動認識チャレンジ」として、qingxinxia@hkust-gz.edu.cnまでお気軽にメールしてください。
# 
# このデータセットとチャレンジについて -> https://abc-research.github.io/challenge2025/
# 
# このノートブックはQingxin Xia, Kei TanigakiとYoshimura Naoyaによって準備されました。

import os
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from data_loader import generate_dataloader, vote_labels, load_data_label
from utils import EarlyStopping
from config import rootdir
from prep import run_fixed_part, div_dataset, load_data
from models.AE_model import CVAE

def train_model(exp_name,train_data, val_data, model_type, num_epochs = 100):
    model = model_type()
    model = model.to(device)

    train_loader = generate_dataloader(train_data, device, if_shuffle=True)
    val_loader = generate_dataloader(val_data, device, if_shuffle=False)

    # ## 3.4 トレーニングとテスト
    # 
    # 参考:
    # https://github.com/jhhuang96/ConvLSTM-PyTorch/blob/master/main.py
    # 

    early_stopping = EarlyStopping()

    learning_rate = 0.001
    optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, amsgrad=True
            )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    fout = open(os.path.join(rootdir,exp_name,'train.log'),'wt')
    def fprint(string):
        print(string)
        fout.write(string+'\n')
        fout.flush()

    fprint(f'{model}')

    permute_left_right = True
    train_losses, val_losses = [], []
    best_val_loss = 10000
    for epoch in range(num_epochs):
        fprint(f'epoch {epoch}/{num_epochs}')
        train_loss, val_loss = [], []
        ###################
        # train the model #
        ###################
        model.train()
        for i, (sample, label) in tqdm(enumerate(train_loader),total=len(train_loader)):
            sample = sample.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.long) # (batchsize,lensw)
            vote_label = vote_labels(label) #(batchsize,1)
            vote_label = vote_label.to(device)
            kwargs = {'labels':vote_label}
            if permute_left_right:
                #同じカテゴリの作業内で左手と右手をそれぞれ別のサンプルとランダムに入れ替える
                for j in range(11):
                    idx = (vote_label == j).nonzero()
                    sample[idx[torch.randperm(len(idx))],:,3:6] = sample[idx,:,3:6]
            loss, output = model(sample,**kwargs)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(np.average(train_loss))
        fprint(f'train loss {np.average(train_loss)}')

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()
            for i, (sample, label) in enumerate(val_loader):
                sample = sample.to(device=device, dtype=torch.float)
                label = label.to(device=device, dtype=torch.long)
                vote_label = vote_labels(label)
                vote_label = vote_label.to(device)
                kwargs = {'labels':vote_label}
                loss, output = model(sample,**kwargs)
                val_loss.append(loss.item())
            avg_val_loss = np.average(val_loss)
            val_losses.append(avg_val_loss)
            fprint(f'val loss {avg_val_loss}')

            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                fprint(f'best val loss improved {best_val_loss} at {epoch}, save the model')
                torch.save(model.state_dict(), os.path.join(rootdir,exp_name,f'AE_model.pth'))
            
            # Check early stopping
            if early_stopping(np.average(val_losses)):
                fprint("Stopping at epoch %s." % str(epoch))
                break
        
        scheduler.step()
        # Print the current learning rate
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        fprint(f'Epoch {epoch + 1}, Learning Rate: {current_lr}')


    plt.figure(figsize=(6,4))
    plt.plot(val_losses, label='valid loss')
    plt.plot(train_losses, label='train loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(rootdir,exp_name,'train_loss.png'))


def test_model(exp_name, test_data, model_type):
    model = model_type()
    model.load_state_dict(torch.load(os.path.join(rootdir,exp_name,'AE_model.pth'), weights_only=True))
    model.to(device)

    test_loader = generate_dataloader(test_data, device, if_shuffle=False)

    fout = open(os.path.join(rootdir,exp_name,'valid.log'),'wt')
    def fprint(string):
        print(string)
        fout.write(string+'\n')
        fout.flush()

    test_loss = []
    with torch.no_grad():
        model.eval()
        for i, (sample, label) in enumerate(test_loader):
            sample = sample.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.long)
            vote_label = vote_labels(label)
            vote_label = vote_label.to(device)
            kwargs = {'labels':vote_label}
            loss, output = model(sample,**kwargs)

            test_loss.append(loss.item())

            if i == 0:
                sample = sample.detach().cpu().numpy()
                output = output.detach().cpu().numpy()

                fig, axs = plt.subplots(2, 1, figsize=(14, 6))
                for i in range(2):
                    sample_ = sample[i,:,:]
                    output_ = output[i,:,:]
                    axs[i].set_title(f'label {i}')
                    axs[i].set_title('Left')
                    axs[i].set_xlabel('timesteps')
                    axs[i].set_ylabel('Value')
                    axs[i].plot(np.arange(0,300), sample_[:,0], label='x')
                    axs[i].plot(np.arange(0,300), sample_[:,1], label='y')
                    axs[i].plot(np.arange(0,300), sample_[:,2], label='z')
                    axs[i].plot(np.arange(0,300), output_[:,0], label='x_re')
                    axs[i].plot(np.arange(0,300), output_[:,1], label='y_re')
                    axs[i].plot(np.arange(0,300), output_[:,2], label='z_re')
                    axs[i].legend()
                plt.tight_layout()
                #plt.show()
                plt.savefig(os.path.join(rootdir,exp_name,'valid_signal.png'))


        fprint(f'Average loss of valid set: {np.average(test_loss):.4f}')

def sample_model(exp_name, model_type):
    model = model_type()
    model.load_state_dict(torch.load(os.path.join(rootdir,exp_name,'AE_model.pth'), weights_only=True))
    model.to(device)

    n_data = 10
    with torch.no_grad():
        model.eval()
        kwargs = {'device':device, 'labels':torch.tensor(np.arange(0,n_data)).to(device)}
        output = model.sample(n_data,**kwargs)

        output = output.detach().cpu().numpy()
        

        fig, axs = plt.subplots(2, 1, figsize=(14, 6))
        for i in range(2):
            output_ = output[i,:,:]
            axs[i].set_title(f'label {i}')
            axs[i].set_xlabel('timesteps')
            axs[i].set_ylabel('Value')
            axs[i].plot(np.arange(0,300), output_[:,0], label='x_re')
            axs[i].plot(np.arange(0,300), output_[:,1], label='y_re')
            axs[i].plot(np.arange(0,300), output_[:,2], label='z_re')
            axs[i].legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(rootdir,exp_name,'test_sample.png'))


def run_cvae1():
    exp_name = expdir+'/pretrain/cvae'
    os.makedirs(exp_name,exist_ok=True)
    train_model(exp_name,train_data, val_data, model_type=CVAE, num_epochs=1000)
    test_model(exp_name, val_data, model_type=CVAE)
    sample_model(exp_name, model_type=CVAE)


if __name__ == '__main__':

    #seed_value, expdir = 2025, 'exp'
    #Training set: ['U2' 'U21' 'U22' 'U3' 'U4' 'U5' 'U6' 'U7']
    #Validation set: ['U1']

    #seed_value, expdir = 40, 'exp2'
    #Training set: ['U1' 'U21' 'U22' 'U3' 'U4' 'U5' 'U6' 'U7']
    #Validation set: ['U2']

    seed_value, expdir = 121, 'exp3'
    #Training set: ['U1' 'U2' 'U21' 'U22' 'U3' 'U5' 'U6' 'U7']
    #Validation set: ['U4']

    selected_columns, new_columns = run_fixed_part(seed_value)

    user_paths, train_users, val_users, test_users = div_dataset()

    train_data_dict, val_data_dict, test_data_dict = load_data(user_paths, train_users, val_users, test_users, selected_columns)

    train_data, val_data, test_data = load_data_label(train_data_dict,val_data_dict,test_data_dict,new_columns,add_data=None)

    run_cvae1()
