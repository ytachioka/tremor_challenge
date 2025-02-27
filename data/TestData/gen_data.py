import pandas as pd
import json
import os
import numpy as np

def gen_data(label_file,test=False):
    label = {}
    subjects = []

    df = pd.read_csv('users_timeXYZ/train_mod.tsv',sep='\t',header=0,index_col=0)
    os.makedirs('real',exist_ok=True)

    stats = {}
    stats['mean'] = {'x':df['x'].mean(),'y':df['y'].mean(),'z':df['z'].mean()}
    stats['max'] = {'x':df['x'].abs().max(),'y':df['y'].abs().max(),'z':df['z'].abs().max()}

    num_seq = {}

    df['x'] -= stats['mean']['x']
    df['y'] -= stats['mean']['y']
    df['z'] -= stats['mean']['z']

    missing = 0

    apply_mvn = True

    data = {}

    with open(label_file) as f:
        i = 0
        for line in f:
            line = line.rstrip()
            if i != 0:
                d = line.split('\t')
                if not test:
                    if d[0] not in label:
                        label[d[0]] = d[1]
                start_t = d[-2]
                end_t = d[-1]
                df_q = df.query(f'{start_t}<= etime <= {end_t}')
                if len(df_q) == 0:
                    missing += 1
                else:
                    ## pickup active part
                    th = {'x':0.3,'y':0.3,'z':0.3}
                    x = df_q['x'].to_numpy()
                    y = df_q['y'].to_numpy()
                    z = df_q['z'].to_numpy()

                    act = np.zeros(len(x),dtype=bool)
                    for i in range(0,len(x),10):
                        act_i = np.std(x[i:min(i+10,len(x))])>=th['x']
                        act_i |= np.std(y[i:min(i+10,len(x))])>=th['y']
                        act_i |= np.std(z[i:min(i+10,len(x))])>=th['z']
                        act[i:min(i+10,len(x))] = act_i

                    subject = d[-3]
                    if subject not in num_seq:
                        num_seq[subject] = 1
                    else:
                        num_seq[subject] += 1

                    df_q = df_q.iloc[act,:]
                    if not test:
                        df_q['label'] = [d[0]]*len(df_q)
                    else:
                        df_q['label'] = [-1]*len(df_q)
                        df_q['seq'] = num_seq[subject]

                    assert len(df_q) > 0

                    if subject not in subjects:
                        subjects.append(subject)
                        data[subject] = df_q
                    else:
                        data[subject] = pd.concat([data[subject],df_q],axis=0)
                    
            i += 1

    for s in subjects:
        stats[s] = {
            'mean':{
                'x':data[s]['x'].mean(),
                'y':data[s]['y'].mean(),
                'z':data[s]['z'].mean()
            },
            'std':{
                'x':data[s]['x'].std(),
                'y':data[s]['y'].std(),
                'z':data[s]['z'].std()
            }
        }
        if apply_mvn:
            data[s]['x'] = (data[s]['x']-stats[s]['mean']['x'])/stats[s]['std']['x']
            data[s]['y'] = (data[s]['y']-stats[s]['mean']['y'])/stats[s]['std']['y']
            data[s]['z'] = (data[s]['z']-stats[s]['mean']['z'])/stats[s]['std']['z']

        data[s].to_csv('real/'+s+'.csv',sep=',')

    print(f'{missing}/{i-1}')

    with open('label.json','wt') as fout:
        json.dump(label,fout,indent=2)

    with open('stats.json','wt') as fout:
        json.dump(stats,fout,indent=2)

if __name__ == '__main__':
    #gen_data('TrainActivities_mod.tsv',False)
    gen_data('TestActivities-20240920_mod.tsv',True)
