import pandas as pd
import json
import os
import numpy as np

def gen_labels(pred_file,label_file):

    df = pd.read_csv(pred_file,sep=',',index_col=0,header=0)

    seq_n = {}
    seq_cnt = {}
    for u in np.unique(df['Subject'].values):
        seq_n[u] = np.max(df[df['Subject'] == u]['seq'].values)
        seq_cnt[u] = 0

    predict_label = {}

    with open(label_file) as f:
        i = 0
        for line in f:
            line = line.rstrip()
            if i != 0:
                d = line.split('\t')
                start_t = float(d[-2])
                end_t = float(d[-1])
                subject = d[-3]
                seq_cnt[subject] += 1
                df_q = df.query(f'Subject == "{subject}" & seq == {seq_cnt[subject]}')
                assert len(df_q) > 0

                ## check etime is correct
                etime = df_q['etime'].values
                assert start_t <= np.min(etime) and np.max(etime) <= end_t

                ## check label is unique
                labels = df_q['label'].values
                label_0 = labels[0]
                for i in range(1,len(labels)):
                    assert label_0 == labels[i], f'{label_0} != {labels[i]}'
                labels = eval(labels[0])

                ids = eval(d[1])
                labels = np.sort(labels[0:len(ids)])
                for id, label in zip(ids,labels):
                    predict_label[id] = str(label)

                 
            i += 1

    ## check all sequences are used
    for u in seq_n.keys():
        assert seq_n[u] == seq_cnt[u]

    with open('label.json','wt') as fout:
        json.dump(predict_label,fout,indent=2)

    return predict_label

def add_labels(test_file,output_file,predict_label):

    with open('data/TrainingDataPD25/label.json') as f:
        label_name = json.load(f)

    with open(test_file) as f:
        with open(output_file,'wt') as fout:
            i = 0
            for line in f:
                d = line.rstrip().split(',')
                if i != 0:
                    d.insert(1,predict_label[d[0]])
                    d.insert(2,label_name[predict_label[d[0]]])
                else:
                    d.insert(1,'Activity Type ID')
                    d.insert(2,'Activity Type')
                                
                i+=1
                fout.write(','.join(d)+'\n')


if __name__ == '__main__':
    predict_label = gen_labels('exp/HAR/virtual_opt_cvae_2model/test.csv','data/TestData/TestActivities-20240920_mod.tsv')
    add_labels('data/TestData/TestActivities-20240920.csv','output.csv',predict_label)
