#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# In[21]:


with open('../pod-validation/data_load.txt') as f:
    data_load = f.readlines()


# In[22]:


cpu_load = {}
mem_load = {}
for i in data_load:
    pod = i.split()[0]
    if pod not in cpu_load:
        cpu_load[pod] = []
        mem_load[pod] = []
        cpu_load[pod].append(int(i.split()[1][:-1]))
        mem_load[pod].append(int(i.split()[2][:-2]))
    else:
        cpu_load[pod].append(int(i.split()[1][:-1]))
        mem_load[pod].append(int(i.split()[2][:-2]))


# In[23]:


name = data_load[0].split()[0]
time = list(range(len(cpu_load[name])))


# In[24]:


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(13,5))

for i in cpu_load.keys():
    ax1.plot(time, cpu_load[i], label='%s' %i)
    ax1.legend(fontsize=12)
ax1.set_xlabel('time(s)',fontsize=16)
ax1.set_ylabel('CPU(m)',fontsize=16)

for i in mem_load.keys():
    ax2.plot(time, mem_load[i], label='%s' %i)
    ax2.legend(fontsize=12)
ax2.set_xlabel('time(s)',fontsize=16)
ax2.set_ylabel('Memory(Mi)',fontsize=16)
fig.suptitle('Loading phase of workload A, Recordcount=2,500,000',fontsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.show()


# In[25]:


with open('../pod-validation/data_run.txt') as f:
    data_run = f.readlines()


# In[26]:


cpu_run = {}
mem_run = {}
for i in data_run:
    pod = i.split()[0]
    if pod not in cpu_run:
        cpu_run[pod] = []
        mem_run[pod] = []
        cpu_run[pod].append(int(i.split()[1][:-1]))
        mem_run[pod].append(int(i.split()[2][:-2]))
    else:
        cpu_run[pod].append(int(i.split()[1][:-1]))
        mem_run[pod].append(int(i.split()[2][:-2]))                 


# In[27]:


name = data_run[0].split()[0]
time = list(range(len(cpu_run[name])))


# In[30]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))

for i in cpu_run.keys():
    ax1.plot(time, cpu_run[i], label='%s' %i)
    ax1.legend(fontsize=12)
ax1.set_xlabel('time(s)',fontsize=16)
ax1.set_ylabel('CPU(m)',fontsize=16)

for i in mem_run.keys():
    ax2.plot(time, mem_run[i], label='%s' %i)
    ax2.legend(fontsize=12)
ax2.set_xlabel('time(s)',fontsize=16)
ax2.set_ylabel('Memory(Mi)',fontsize=16)
fig.suptitle('Running phase of workload A, Operationcount=2,500,000',fontsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.show()


# In[ ]:




