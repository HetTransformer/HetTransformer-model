#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm


# In[2]:


fnn_pathway = ''
output_dir = ''


# In[ ]:


news_label = list()
for dataset in ['politifact/', 'gossipcop/']:
    print('reading %s...' %dataset)
    pathway = fnn_pathway + dataset
    for post_type in ['fake/', 'real/']:
        print('reading post info from %s file...' %post_type)
        if post_type == 'fake/':
            label = '0'
        else:
            label = '1'
        news_list = os.listdir(pathway + post_type)
        print("%d news from %s" %(len(news_list), dataset))
        for n in news_list:
            news_label.append([n, label])

with open(output_dir + 'news_label.txt', 'w') as f:
    for news in news_label:
        f.write(' '.join(news) + '\n')

