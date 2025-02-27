import optuna
import numpy as np
import subprocess
import os
import shutil
import random
import joblib

r_type = 2
exp_name = f'virtual_opt_cvae_2model'

def objective(trial):
    r_accum = 0
    r_ = [0]*10

    num_classes = 10

    if r_type == 1:
        min_p = 0.05
        max_p = 0.25
        for i,j in enumerate(random.sample(range(num_classes-1), k=num_classes-1)):
            assert 1>r_accum, r_accum
            r_j = trial.suggest_float(f'r{j}',0,min(max_p,1-min_p*(num_classes-1-i)-r_accum))
            r_accum += r_j
            r_[j] = r_j
        r = [0]
        r_accum = 0
        for i in range(num_classes-1):
            r_accum += r_[i]
            r.append(r_accum)
        r.append(1)
    elif r_type == 2:
        c = [0]*num_classes
        for i in range(num_classes):
            c[i] = trial.suggest_int(f'c{i}',0,100)
        r_accum = 0
        r = [0]
        for i in range(num_classes-1):
            r_i = c[i]/sum(c)
            r_accum += r_i
            r.append(r_accum)
        r.append(1)

    sampling_ratio = trial.suggest_float('sampling_ratio',0.1,0.85) 
    step = trial.suggest_int('step',10,300)

    v_dir = os.path.join('data',exp_name)
    if os.path.isdir(v_dir):
        shutil.rmtree(v_dir)
    os.makedirs(v_dir)
    print(np.array(r))
    np.save(os.path.join(v_dir,'ratio.npy'), np.array(r))
    np.save(os.path.join(v_dir,'params.npy'), np.array([sampling_ratio,step]))
    subprocess.call(['python', '3_gen_data_prop.py'])
    subprocess.call(['python', '2_train_evaluateHAR.py'])


    with open(os.path.join(exp_dir,'train.log'),'rt') as f:
        for i in f:
            i = i.rstrip()
            if i.startswith('best F1 (val):'):
                acc = float(i.split(': ')[1])
                break

    exp1_dir = os.path.join(exp_dir,str(trial.number))
    os.makedirs(exp1_dir)
    for f in ['train_cm.png','valid_cm.png','test.log','train_F1.png','train.log','train_loss.png']:
        shutil.move(os.path.join(exp_dir,f),os.path.join(exp1_dir,f))
    os.remove(os.path.join(exp_dir,'HAR_model.pth'))

    return acc

#exp_dir = os.path.join('exp','HAR',exp_name)
#exp_dir = os.path.join('exp2','HAR',exp_name)
exp_dir = os.path.join('exp3','HAR',exp_name)

if os.path.isdir(exp_dir):
    shutil.rmtree(exp_dir)
os.makedirs(exp_dir)

study = optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=100)

study.trials_dataframe().to_csv(os.path.join(exp_dir,'run_hist.csv'))
joblib.dump(study, os.path.join(exp_dir,'study.pkl'))

fig = optuna.visualization.plot_param_importances(study)
fig.write_image(os.path.join(exp_dir,'param_importance.png'))

fig = optuna.visualization.plot_optimization_history(study)
fig.write_image(os.path.join(exp_dir,'optimization_history.png'))

fig = optuna.visualization.plot_contour(study)
fig.write_image(os.path.join(exp_dir,'contour.png'))


trial = study.best_trial
print("Best Trial#{}".format(trial.number))
print("  Params: {}".format(trial.params))
