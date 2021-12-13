#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymongo
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from kubernetes import client,config,watch


# In[2]:


mongoclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongoclient['resource-v1']


# In[3]:


collist = db.list_collection_names()
collist


# In[4]:


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


# In[5]:


record = {}
for col in collist:
    record[col] = pd.DataFrame(list(db[col].find()))
    # convert timestamp to time
    record[col]['time'] = pd.to_datetime(record[col]['time'],unit='s').round('1s')
    record[col]['cpu'] = record[col]['cpu'].apply(lambda x: convert_cpu(x))
    record[col]['cpu'] = record[col]['cpu'].fillna(0)
    record[col]['memory'] = record[col]['memory'].apply(lambda x: convert_mem(x))


# In[6]:


request = record['requests']
usage = record['usage']
request = request[request['pod_name'] != 'ycsb']
usage = usage[usage['pod_name'] != 'ycsb']
usage.head()


# In[7]:


pod_usage = {}
for i in usage['pod_name'].unique():
    pod_usage[i] = usage[usage['pod_name'] == i]
    
pod_request = {}
for i in request['pod_name'].unique():
    pod_request[i] = request[request['pod_name'] == i]
print('len(pod_usage) =',len(pod_usage))
print('len(pod_request) =',len(pod_request))


# In[8]:


fig, ax = plt.subplots(len(pod_request),1,figsize=(13,8),sharex=True,sharey=True)
for index,pod in enumerate(pod_request):
    ax[index].scatter(pod_request[pod].time, pod_request[pod].cpu,marker='.',label=pod +' request')
    ax[index].plot(pod_usage[pod].time, pod_usage[pod].cpu,label=pod +' usage',color='r')
    ax[index].legend(fontsize=16)
    ax[index].set_ylabel('CPU(m)', fontsize=16)
    ax[index].set_xlabel('timeline', fontsize=16)
    ax[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax[index].xaxis.set_minor_locator(mdates.MinuteLocator())
fig.autofmt_xdate() #rotate labels
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.show()


# In[11]:


fig, ax = plt.subplots(len(pod_request),1,figsize=(13,8),sharex=True,sharey=True)
for index,pod in enumerate(pod_request):
    ax[index].scatter(pod_request[pod].time, pod_request[pod].memory,marker='.',label=pod +' request')
    ax[index].plot(pod_usage[pod].time, pod_usage[pod].memory,label=pod +' usage',color='r')
    ax[index].legend(fontsize=16)
    ax[index].set_ylabel('Memory(Mi)', fontsize=16)
    ax[index].set_xlabel('timeline', fontsize=16)
    ax[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax[index].xaxis.set_minor_locator(mdates.MinuteLocator())
fig.autofmt_xdate() #rotate labels
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.show()


# In[ ]:




