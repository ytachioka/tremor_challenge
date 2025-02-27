import datetime
from datetime import datetime as dt
import locale

locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')

dt1 = dt(year=2024, month=9, day=1, hour=0)

tmp_rep = [
    ("9 Walk & STOP/frozen, full body shaking, rotate then return back","9 Walk & STOP/frozen- full body shaking- rotate then return back"),
    ("10 Slow walk (SHAKING hands/body, tiny step, head forward)","10 Slow walk (SHAKING hands/body- tiny step- head forward)")
]

with open('TrainActivities.csv') as f:
    with open('TrainActivities_mod.tsv','wt') as fout:
        prev_line = ''
        i = 0
        for line in f:
            for t in tmp_rep:
                line = line.replace(t[0],t[1])
            d = line.rstrip().split(',')
            if d[3] == '':
                d[3] = d[5]
            if d[4] == '':
                d[4] = d[5]
            if i != 0:
                # 2024/09/02 6:16
                d[3] = dt.strptime(d[3],'%Y/%m/%d %H:%M')
                d[4] = dt.strptime(d[4],'%Y/%m/%d %H:%M')+datetime.timedelta(seconds=59)
                d.append(int((d[3]-dt1).total_seconds()))
                d.append(int((d[4]-dt1).total_seconds()))
                d[3],d[4] = str(d[3]), str(d[4])
                d[7],d[8] = str(d[7]), str(d[8])
            else:
                d.append('elapsed time (start)')
                d.append('elapsed time (end)')
            
            for t in tmp_rep:
                d[2] = d[2].replace(t[1],t[0])
            d.pop(0)
            print(d)
            line = '\t'.join(d)
            if prev_line != line:
                fout.write(line)
                fout.write('\n')
            prev_line = line
            i+=1

