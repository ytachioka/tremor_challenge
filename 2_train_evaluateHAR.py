#!/usr/bin/env python
# coding: utf-8

## traing and evaluate HAR model with data augmentation for tremor challenge

import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from data_loader import generate_dataloader, vote_labels, load_data_label
from config import rootdir
from prep import run_fixed_part, div_dataset, load_data
from models.HAR_model import Transformer
from utils import EarlyStopping

def draw_cm(y_true,y_pred,filename):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, cmap = 'Blues', annot=True, fmt = 'd')
    plt.yticks(rotation=0)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# # to improve HAR model performance by using generated data
# https://github.com/jhhuang96/ConvLSTM-PyTorch/blob/master/main.py

def train_model(exp_name,train_data, val_data, model_type = Transformer, num_epochs = 200):
    model = model_type()
    model = model.to(device)
    print(model)

    train_loader = generate_dataloader(train_data, device, if_shuffle=True)
    val_loader = generate_dataloader(val_data, device, if_shuffle=False)

    early_stopping = EarlyStopping(patience=15,direction='maximize')

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

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


    train_losses, val_losses = [], []
    train_F1, val_F1 = [], []
    best_val_loss = 10000
    best_F1 = 0
    for epoch in range(num_epochs):
        fprint(f'epoch {epoch}/{num_epochs}')
        train_loss, val_loss = [], []
        ###################
        # train the model #
        ###################
        model.train()
        true_labels, pred_labels = [], []
        for i, (sample, label) in tqdm(enumerate(train_loader),total=len(train_loader)):
            sample = sample.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.long) # (batchsize,lensw)
            vote_label = vote_labels(label) #(batchsize,1)
            vote_label = vote_label.to(device)
            output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
            loss = criterion(output, vote_label)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            true_labels.append(vote_label.detach().cpu().numpy())
            pred_labels.append(output.detach().cpu().numpy())

        train_losses.append(np.average(train_loss))
        fprint(f'train loss {np.average(train_loss)}')
        # Calculate F1 scores
        y_true = np.concatenate(true_labels, axis=0)
        y_prob = np.concatenate(pred_labels, axis=0)

        # Get the predicted class labels (argmax along the class dimension)
        y_pred = np.argmax(y_prob, axis=1)  # output Shape: (batch_size, time_steps)

        # Calculate F1 score (macro F1 score)
        f1 = f1_score(y_true, y_pred, average='macro')
        train_F1.append(f1)
        fprint(f'F1 Score of training set: {f1:.4f}')

        draw_cm(y_true,y_pred,os.path.join(rootdir,exp_name,'train_cm.png'))

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()
            true_labels, pred_labels = [], []
            for i, (sample, label) in enumerate(val_loader):
                sample = sample.to(device=device, dtype=torch.float)
                label = label.to(device=device, dtype=torch.long)
                vote_label = vote_labels(label)
                vote_label = vote_label.to(device)
                output = model(sample)
                loss = criterion(output, vote_label)
                val_loss.append(loss.item())
                true_labels.append(vote_label.detach().cpu().numpy())
                pred_labels.append(output.detach().cpu().numpy())
            avg_val_loss = np.average(val_loss)
            val_losses.append(avg_val_loss)
            fprint(f'val loss {avg_val_loss}')

            # Calculate F1 scores
            y_true = np.concatenate(true_labels, axis=0)
            y_prob = np.concatenate(pred_labels, axis=0)

            # Get the predicted class labels (argmax along the class dimension)
            y_pred = np.argmax(y_prob, axis=1)  # output Shape: (batch_size, time_steps)

            # Calculate F1 score (macro F1 score)
            f1 = f1_score(y_true, y_pred, average='macro')
            val_F1.append(f1)
            fprint(f'F1 Score of validation set: {f1:.4f}')

            '''
            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                fprint(f'best val loss improved {best_val_loss}, save the model')
                torch.save(model.state_dict(), os.path.join(rootdir,exp_name,'HAR_model.pth'))
            '''

            if best_F1 < f1:
                best_F1 = f1
                draw_cm(y_true,y_pred,os.path.join(rootdir,exp_name,'valid_cm.png'))
                fprint(f'best F1 improved {best_F1}, save the model')
                torch.save(model.state_dict(), os.path.join(rootdir,exp_name,'HAR_model.pth'))

            # Check early stopping
            #if early_stopping(np.average(val_losses)):
            if early_stopping(f1):
                fprint("Stopping at epoch %s." % str(epoch))
                break
        
        scheduler.step()
        # Print the current learning rate
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        fprint(f'Epoch {epoch + 1}, Learning Rate: {current_lr}')
    
    fprint(f'best F1 (val): {best_F1}')


    plt.figure(figsize=(6,4))
    plt.plot(val_losses, label='valid loss')
    plt.plot(train_losses, label='train loss')
    plt.grid()
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(rootdir,exp_name,'train_loss.png'))

    plt.figure(figsize=(6,4))
    plt.plot(val_F1, label='valid F1')
    plt.plot(train_F1, label='train F1')
    plt.grid()
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('F1')
    plt.ylim(0,1)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(rootdir,exp_name,'train_F1.png'))


def test_model(exp_name, test_data, model_type):
    test_loader = generate_dataloader(test_data, device, if_shuffle=False)
    model = model_type()
    model.load_state_dict(torch.load(os.path.join(rootdir,exp_name,'HAR_model.pth'), weights_only=True))
    model = model.to(device)

    fout = open(os.path.join(rootdir,exp_name,'test.log'),'wt')
    def fprint(string):
        print(string)
        fout.write(string+'\n')
        fout.flush()
    
    with torch.no_grad():
        model.eval()
        true_labels, pred_labels = [], []
        for i, (sample, label) in enumerate(test_loader):
            sample = sample.to(device=device, dtype=torch.float)
            #label = label.to(device=device, dtype=torch.long)
            #vote_label = vote_labels(label)
            vote_label = torch.tensor([0]*sample.shape[0])
            # vote_label = vote_label.to(device)
            output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13

            true_labels.append(vote_label.numpy())
            pred_labels.append(output.detach().cpu().numpy())

        # Calculate F1 scores
        y_true = np.concatenate(true_labels, axis=0)
        y_prob = np.concatenate(pred_labels, axis=0)

        # Get the predicted class labels (argmax along the class dimension)
        y_pred = np.argmax(y_prob, axis=1)  # output Shape: (batch_size, time_steps)

        # Calculate F1 score (macro F1 score)
        f1 = f1_score(y_true, y_pred, average='macro')

        fprint(f'F1 Score of test set: {f1:.4f}')

        draw_cm(y_true,y_pred,os.path.join(rootdir,exp_name,'test_cm.png'))


def run_baseline1():
    exp_name = expdir+'/HAR/real_only'

    train_data, val_data, test_data = load_data_label(train_data_dict,val_data_dict,test_data_dict,new_columns,add_data=None)
    os.makedirs(exp_name,exist_ok=True)

    train_model(exp_name,train_data, val_data, model_type=Transformer)
    test_model(exp_name, test_data, model_type=Transformer)

def run_baseline2(virt='virtual'):
    ## just add virtual data
    exp_name = expdir+'/HAR/'+virt
    add_data=os.path.join('data',exp_name.split(os.sep)[-1])

    train_data, val_data, test_data = load_data_label(train_data_dict,val_data_dict,test_data_dict,new_columns,add_data=add_data)

    os.makedirs(exp_name,exist_ok=True)

    train_model(exp_name,train_data, val_data, model_type=Transformer)
    test_model(exp_name, test_data, model_type=Transformer)



if __name__ == '__main__':

    #seed_value, expdir = 2025, 'exp'
    #seed_value, expdir = 40, 'exp2'
    seed_value, expdir = 121, 'exp3'
    selected_columns, new_columns = run_fixed_part(seed_value)

    user_paths, train_users, val_users, test_users = div_dataset()
    train_data_dict, val_data_dict, test_data_dict = load_data(user_paths, train_users, val_users, test_users, selected_columns)

    #run_baseline1()
    #run_baseline2()
    #run_baseline2('virtual_vae')
    #run_baseline2('virtual_cvae')
    #run_baseline2('virtual_cvae_2model')
    run_baseline2('virtual_opt_cvae_2model')


