import numpy as np
import subprocess
import os
import shutil
import pandas as pd

r_type = 2
exp_name = f'virtual_opt_cvae_2model'

def doexp(df):

    idx = df['value'].idxmax()

    print(f'best trial is {idx}')

    r_accum = 0
    num_classes = 10

    if r_type == 1:
        assert 0
        r_ = [0]*(num_classes-1)
        min_p = 0.05
        max_p = 0.25
        #for i,j in enumerate(random.sample(range(num_classes-1), k=num_classes-1)):
        #    assert 1>r_accum, r_accum
        #    r_j = trial.suggest_float(f'r{j}',0,min(max_p,1-min_p*(num_classes-1-i)-r_accum))
        #    r_accum += r_j
        #    r_[j] = r_j
        r = [0]
        r_accum = 0
        for i in range(num_classes-1):
            r_accum += r_[i]
            r.append(r_accum)
        r.append(1)
    elif r_type == 2:
        c = df.iloc[idx,5:(5+num_classes)].values
        r_accum = 0
        r = [0]
        for i in range(num_classes-1):
            r_i = c[i]/sum(c)
            r_accum += r_i
            r.append(r_accum)
        r.append(1)

    sampling_ratio = df.iloc[idx,:]['params_sampling_ratio']
    step = df.iloc[idx,:]['params_step']

    v_dir = os.path.join('data',exp_name)
    if os.path.isdir(v_dir):
        shutil.rmtree(v_dir)
    os.makedirs(v_dir)
    print(np.array(r))
    np.save(os.path.join(v_dir,'ratio.npy'), np.array(r))
    np.save(os.path.join(v_dir,'params.npy'), np.array([sampling_ratio,step]))
    subprocess.call(['python', '3_gen_data_prop.py'])
    subprocess.call(['python', '2_train_evaluateHAR.py'])

    return 0

exp_dir = os.path.join('exp','HAR',exp_name)
#exp_dir = os.path.join('exp2','HAR',exp_name)

df = pd.read_csv(os.path.join(exp_dir,'run_hist.csv'),index_col=0,header=0,sep=',')
doexp(df)
