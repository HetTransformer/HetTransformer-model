#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import os
import numpy as np
from tqdm import tqdm




# In[6]:


fnn_pathway = ''
output_dir = ''


# In[ ]:




for dataset in ['politifact/', 'gossipcop/']:
    print('reading %s...' %dataset)
    pathway = fnn_pathway + dataset
    for post_type in ['fake/', 'real/']:
        print('reading post info from %s file...' %post_type)
        news_list = os.listdir(pathway + post_type)

        pid = list()
        post_features = list()
        for file in tqdm(news_list, desc='reading each news file'):
            if not os.path.exists(pathway + post_type + file + '/tweets'):
                continue;
            tweets_list = os.listdir(pathway + post_type + file + '/tweets')
            for tweet_file in tweets_list:
                try:
                    with open(pathway + post_type + file + '/tweets/%s' %tweet_file) as f: # read json files
                        data = json.load(f)
                except:
                    continue
                    
                        
                pid.append(data['id_str'])
                num_retweet = str(data['retweet_count'])
                    
                # skip num word description
                # skip num word name
                num_favorite = str(data['favorite_count'])
                is_quote_status = str(data['is_quote_status']-0)

                post_features.append([num_retweet, num_favorite, is_quote_status])
                
            if not os.path.exists(pathway + post_type + file + '/retweets'):
                continue;
            tweets_list = os.listdir(pathway + post_type + file + '/retweets')
            for tweet_file in tweets_list:
                try:

                    with open(pathway + post_type + file + '/retweets/%s' %tweet_file) as f: # read json files
                        data = json.load(f)
                except:
                    continue
     
                for t in data['retweets']:

                    pid.append(t['id_str'])
                    num_retweet = str(t['retweet_count'])

                    # skip num word description
                    # skip num word name
                    num_favorite = str(t['favorite_count'])
                    is_quote_status = str(t['is_quote_status']-0)

                    post_features.append([num_retweet, num_favorite, is_quote_status])

    
with open(output_dir + 'FNN_post_features.txt', 'w') as f:
    for i in tqdm(range(len(pid)), 'writing post features'):
        f.write('p%s: ' %pid[i])
        f.write(' '.join(post_features[i]))
        f.write('\n')
    
   



