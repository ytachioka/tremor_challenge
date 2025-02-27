import pandas as pd
import datetime
from datetime import datetime as dt
import locale

locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')

dt1 = dt(year=2024, month=9, day=1, hour=0)

data = {
'ID':[],
'Started':[],
'Finished':[],
'Updated':[],
'Subject':[],
'elapsed time (start)':[],
'elapsed time (end)':[]
}


with open('TestActivities-20240920.csv') as f:
    prev_line = ''
    i = 0
    for line in f:
        d = line.rstrip().split(',')
        if d[1] == '':
            d[1] = d[3]
        if d[2] == '':
            d[2] = d[3]

        line = [d[1],d[2],d[4]]

        if i != 0 and line != prev_line:
            # 2024/09/02 6:16
            d[1] = dt.strptime(d[1],'%Y/%m/%d %H:%M')
            d[2] = dt.strptime(d[2],'%Y/%m/%d %H:%M')+datetime.timedelta(seconds=59)
            d[3] = dt.strptime(d[3],'%Y/%m/%d %H:%M')
            data['ID'].append([d[0]])
            data['Started'].append(d[1])
            data['Finished'].append(d[2])
            data['Updated'].append(d[3])
            data['Subject'].append(d[4])
            data['elapsed time (start)'].append(int((d[1]-dt1).total_seconds()))
            data['elapsed time (end)'].append(int((d[2]-dt1).total_seconds()))
        elif i != 0:
            data['ID'][-1].append(d[0])
                        
        prev_line = line
        i+=1

pd.DataFrame.from_dict(data).to_csv('TestActivities-20240920_mod.tsv',sep='\t')
