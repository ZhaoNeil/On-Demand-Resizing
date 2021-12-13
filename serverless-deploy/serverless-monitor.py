#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import json
import pymongo
import pandas as pd
from datetime import datetime
from kubernetes import client,config,watch


# In[2]:


config.load_kube_config()
usage_api = client.CustomObjectsApi()
request_api = client.CoreV1Api()
vpa_api = client.ApiClient()


# In[3]:


mongoclient = pymongo.MongoClient("mongodb://localhost:27017/")
mongoclient.drop_database('video_processing_67m_sma5-3')
db = mongoclient['video_processing_67m_sma5-3']


# In[4]:


for i in range(600):
    timeline = datetime.timestamp(datetime.now())
    
    #usage
    usage = usage_api.list_namespaced_custom_object(group="metrics.k8s.io",version="v1beta1", namespace="default", plural="pods")['items']
    for i in usage:
        collection = db['usage']
        item = dict({'time':timeline}, **i['containers'][0]['usage'])
        item.update({'pod_name':i['metadata']['name']})
        collection.insert_one(item)
    
    #request    
    request = request_api.list_namespaced_pod(namespace='default').items
    for i in request:
        collection = db['requests']
        if i.spec.containers[0].resources.requests != None:
            item = dict({'time':timeline}, **i.spec.containers[0].resources.requests)
            item.update({'pod_name':i.metadata.name})
            collection.insert_one(item)
        else:
            item = dict({'time':timeline}, **{'cpu':0, 'memory':0})
            item.update({'pod_name':i.metadata.name})
            collection.insert_one(item)
    
    #vpa recommendation
    collection = db['vpa']
    vpa_metrics = vpa_api.call_api('/apis/autoscaling.k8s.io/v1/namespaces/default/verticalpodautoscalers/redis-vpa', 'GET', _preload_content=False) 
    vpa_metrics = vpa_metrics[0].data.decode('utf-8')
    recommendation = json.loads(vpa_metrics)
    target = recommendation['status']['recommendation']['containerRecommendations'][0]['target']
    lowerBound = recommendation['status']['recommendation']['containerRecommendations'][0]['lowerBound']
    upperBound = recommendation['status']['recommendation']['containerRecommendations'][0]['upperBound']
    d1 = {'target_cpu':target['cpu'], 'target_mem':target['memory']}
    d2 = {'lowerBound_cpu':lowerBound['cpu'], 'lowerBound_mem':lowerBound['memory']}
    d3 = {'upperBound_cpu':upperBound['cpu'], 'upperBound_mem':upperBound['memory']}
    d4 = {'containerName':recommendation['status']['recommendation']['containerRecommendations'][0]['containerName']}
    item = {'time':timeline}
    for d in d1,d2,d3,d4:
        item.update(d)
    collection.insert_one(item)
    
    time.sleep(1)


# In[5]:


# data = usage_api.list_namespaced_custom_object(group="metrics.k8s.io",version="v1beta1", namespace="default", plural="pods")['items']
# for i in data:
#     print(i['metadata']['name'])


# In[6]:


# data[0]


# In[7]:


# pod = request_api.list_namespaced_pod(namespace='default')
# print(pod.items[0].metadata.name)
# print(pod.items[0].spec.containers[0])


# In[8]:


# pod.items[1].spec.containers[0]


# In[9]:


# vpa_metrics = vpa_api.call_api('/apis/autoscaling.k8s.io/v1/namespaces/default/verticalpodautoscalers/redis-vpa', 'GET', _preload_content=False) 
# vpa_metrics = vpa_metrics[0].data.decode('utf-8')
# recommendation = json.loads(vpa_metrics)
# recommendation['status']['recommendation']['containerRecommendations']

