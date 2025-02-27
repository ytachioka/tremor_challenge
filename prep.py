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
import random
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from config import realpath, virtpath, rootdir, real_directory, real_directory_test, virt_directory, splits


# ## 1.3 修正不可能な箇所
# 
# 参加者は、splitとrandom seedを変更して、アルゴリズムの堅牢性を測定できます。我々がコードを評価する際には、selected_columns と new_columns は変更されませんが、split と random seed は変更されます。
# 
def run_fixed_part(seed_value=2025):
    print('Randomly Split the real dataset into train, validation and test sets: %s'%str(splits))

    selected_columns = ['x','y','z','label']
    print('Select acceleration data of both wrists: %s'%selected_columns)

    new_columns = selected_columns[:3] + [selected_columns[-1]]
    print('Data for train, validation, and test: %s'%new_columns)

    def set_random_seed(seed):
        # Set seed for Python's random module
        random.seed(seed)

        # Set seed for NumPy
        np.random.seed(seed)

        # Set seed for PyTorch
        torch.manual_seed(seed)

        # If using CUDA, set seed for GPU as well
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set a fixed random seed
    set_random_seed(seed_value)

    return selected_columns, new_columns


# # 2. 実データを使用して仮想データを生成する
# 
# まず、実データを特定の比率に従ってトレーニングセット、検証セット、テストセットにランダムに分割します。参加者はトレーニングセットを使用して仮想データを生成できます。最後に、トレーニングセットと仮想データの両方を使用してネットワークをトレーニングし、テストセットの F1 スコアを計算します。
# 
# 次のコードは、トレーニング セットから仮想データを生成する例を示しています。次の点に注意してください。
# 
# (1) モデル構造は固定されており、変更することはできません。
# 
# (2) トレーニングセットとテストセットの分割比率、およびランダムシードは現在の設定とは異なります。そのため、仮想データ生成アルゴリズムは変化するデータに対して堅牢である必要があります。
# 
# (3) 参加者はデータ生成アルゴリズムを自由に設計し、指定されたパス「/data/virtual/」に保存できますが、仮想データのサイズは500MBに制限されます。

def div_dataset():
    # ## 2.1 トレーニング ユーザー、検証ユーザー、テスト ユーザーを割り当てる
    # OpenPack データセットでは、U0xxx はユーザー ID に対応し、S0xxx はさまざまな実験設定に対応します。
    # 
    # このチャレンジでは、S0100 からトレーニング (実際の) データとテスト データのみを選択します。
    # 
    # ## 2.2 未使用のデータを除外する
    user_paths = {}
    for root, dirs, files in os.walk(real_directory):
        print(files)
        for file in files:
            user_paths[file[:-4]] = os.path.join(root, file)
    #for u, d in user_paths.items():
    #    print('%s at: %s'% (u,d))
    
    # ## 2.3 ユーザーをトレーニング、検証、テストセットに分割する
    userIDs = list(user_paths.keys())

    # Shuffle the list to ensure randomness
    random.shuffle(userIDs)

    # Calculate the split indices
    total_length = len(userIDs)
    train_size = int(total_length * splits[0])
    val_size = int(total_length * splits[1])

    # Split the list according to the calculated sizes
    train_users = np.sort(userIDs[:train_size])      # First 70%
    val_users = np.sort(userIDs[train_size:])  # Next 10%

    user_paths_test = {}
    for root, dirs, files in os.walk(real_directory_test):
        for file in files:
            user_paths_test[file[:-4]] = os.path.join(root, file)
    #for u, d in user_paths_test.items():
    #    print('%s at: %s'% (u,d))

    test_users = np.sort(list(user_paths_test.keys()))
    user_paths |= user_paths_test

    print('Training set: %s'%train_users)
    print('Validation set: %s'%val_users)
    print('Test set: %s'%test_users)

    return user_paths, train_users, val_users, test_users


def load_data(user_paths, train_users, val_users, test_users,selected_columns):
    # ## 2.4 ユーザーID に従ってデータをロードします
    # 
    # すべてのユーザーのデータをデータフレームとしてロードします。
    # 両手首の加速度データのみを使用します。
    # 操作ラベルを使用します。
    # 
    train_data_dict = {}
    for u in train_users:
        # Load the CSV file with only the selected columns
        train_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

    val_data_dict = {}
    for u in val_users:
        # Load the CSV file with only the selected columns
        val_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

    test_data_dict = {}
    selected_columns_test = selected_columns.copy()
    selected_columns_test.append('seq')
    for u in test_users:
        # Load the CSV file with only the selected columns
        test_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns_test)

    return train_data_dict, val_data_dict, test_data_dict

def plot_data(user_name):
    # ## 2.5 データの例を示す
    # 
    df = train_data_dict[user_name]

    n = 10  # only show n timestamps on fig
    # timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
    # dates = [str(datetime.datetime.fromtimestamp(ts / 1000).replace(tzinfo=timezone_jst)) for ts in df['timestamp'].values]
    dates = df.timestamp.values
    # Select n equally spaced indices to show on the x-axis
    indices = np.linspace(0, len(dates) - 1, n, dtype=int)
    selected_dates = [dates[i] for i in indices]

    data = df[['atr01/acc_x','atr01/acc_y','atr01/acc_z']].values

    l = df['operation'].values


    fig, axs = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle(user_name)
    # First subplot
    axs[0].plot(dates, data[:,0], label='x')
    axs[0].plot(dates, data[:,1], label='y')
    axs[0].plot(dates, data[:,2], label='z')
    axs[0].set_title('Raw data')
    axs[0].set_xlabel('timesteps')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    # Set x-ticks for the current subplot
    # axs[0].set_xticks(selected_dates)
    # axs[0].set_xticklabels(selected_dates, rotation=60)  # Set labels and rotate
    # axs[0].set_xlim([dates[0], dates[-1]])  # Set x-axis limits
    # axs[0].grid()

    # Second subplot
    axs[1].plot(dates, l, label='label value')
    axs[1].set_title('Operation labels')
    axs[1].set_xlabel('timesteps')
    axs[1].set_ylabel('Label ID')
    axs[1].set_xticks(selected_dates)
    axs[1].set_xticklabels(selected_dates, rotation=30)  # Set labels and rotate
    axs[1].set_xlim([dates[0], dates[-1]])  # Set x-axis limits
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plots
    #plt.show()
    plt.savefig(rootdir+'/'+user_name+'.png')

def get_folder_size(folder_path):
    # ##2.9 仮想データのサイズを確認する
    # 
    # このコードを使用して、仮想データのサイズが 500MB を超えていないことを確認します。
    # 
    total_size = 0
    # Walk through all files and subdirectories in the folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # Get the full file path
            file_path = os.path.join(dirpath, filename)
            # Add the size of each file to the total size
            total_size += os.path.getsize(file_path)
    return total_size


if __name__ == '__main__':

    selected_columns, new_columns = run_fixed_part()

    if not os.path.exists('data/real/imu-with-operation-action-labels/imu-with-operation-action-labels/U0101-S0100.csv'):
        prep()

    user_paths, train_users, val_users, test_users = div_dataset()
    train_data_dict, val_data_dict, test_data_dict = load_data(user_paths, train_users, val_users, test_users, selected_columns)

    plot_data(train_users[0])


