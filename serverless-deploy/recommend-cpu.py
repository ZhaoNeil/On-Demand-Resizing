#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymongo
import math
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates


# In[9]:


mongoclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongoclient['video_processing_17m_ema5-3']
db.name


# In[10]:


# convert cpu unit 'n' to 'm'
def convert_cpu(x):
    if x != 0:
        if x[-1] == 'n':
            return round(int(x[:-1])/1000000)
        elif x[-1] == 'm':
            return int(x[:-1])
    else:
        return 0
    
# convert memory unit 'k' and 'Ki' to 'Mi'
# 1k = 1000/1024 Ki = 1000/1024**2 Mi
def convert_mem(x):
    if x != 0:
        if x[-1] == 'k':
            return round(int(x[:-1])*1000/1024**2)
        elif x[-2:] == 'Ki':
            return round(int(x[:-2])/1024)
        elif x[-2:] == 'Mi':
            return int(x[:-2])
    else:
        return 0


# In[11]:


collist = db.list_collection_names()
vpa = pd.DataFrame(list(db['vpa'].find()))
vpa['time'] = pd.to_datetime(vpa['time'],unit='s').round('1s')
record = {}
collist = ['usage', 'requests']
for col in collist:
    record[col] = pd.DataFrame(list(db[col].find()))
    # convert timestamp to time
    record[col]['time'] = pd.to_datetime(record[col]['time'],unit='s').round('1s')
    record[col]['cpu'] = record[col]['cpu'].apply(lambda x: convert_cpu(x))
    record[col]['cpu'] = record[col]['cpu'].fillna(0)
vpa['target_cpu'] = vpa['target_cpu'].apply(lambda x: convert_cpu(x))
usage = record['usage']
usage = usage[usage['pod_name'] == usage.iloc[0]['pod_name']]
usage=usage[['time','cpu']]
vpa = vpa[['time','target_cpu']]
result = pd.merge(usage, vpa, how='right', on="time")
result['cpu'] = result['cpu'].fillna(0)


# In[12]:


fig, ax = plt.subplots(1,1,figsize=(13,8),sharex=True,sharey=True)
ax.plot(result.time, result.cpu,label='CPU usage',color='r')
ax.plot(result.time, result.target_cpu,label=db.name,color='b')
ax.legend(fontsize=16)
ax.set_ylabel('CPU(m)', fontsize=16)
ax.set_xlabel('timeline', fontsize=16)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.xaxis.set_minor_locator(mdates.MinuteLocator())
fig.autofmt_xdate() #rotate labels
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.show()


# In[6]:


def evaluate(usage, prediction):
    slack = list(map(operator.sub, prediction, usage))
    slack_total = 0
    insufficient_count = 0
    insufficient_total = 0
    for i in slack:
        if i < 0:
            insufficient_count += 1
            insufficient_total += abs(i)
        else:
            slack_total += i
    return slack_total/len(usage), insufficient_count/len(usage)*100, insufficient_total/len(usage)


# In[7]:


db_name = []
db_name = evaluate(usage.cpu, vpa.target_cpu)
print("\t\tAvg. slack(m)\t Insufficient CPU(%)\t Avg. insufficient CPU(m)")
print(db.name, db_name[0], db_name[1], db_name[2])


# In[8]:


# with open ('ycsb.txt','a') as f:
#     f.write(db.name+'\t'+str(db_name[0])+'\t'+str(db_name[1])+'\t'+str(db_name[2])+'\n')


# In[ ]:




