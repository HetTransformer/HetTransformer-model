#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import os
import pandas as pd
from tqdm import tqdm


fnn_pathway = ''
output_dir = ''



uid = list()
user_features = list()

for dataset in ['politifact/', 'gossipcop/']:
    print('reading %s...' %dataset)
    pathway = fnn_pathway + dataset
    for post_type in ['fake/', 'real/']:
        
        print('reading user info from %s file...' %post_type)
        news_list = os.listdir(pathway + post_type)
        
        for file in tqdm(news_list, desc='reading each news'):
            if not os.path.exists(pathway + post_type + file + '/tweets'):
                continue;
            tweets_list = os.listdir(pathway + post_type + file + '/tweets')
            for tweet_file in tweets_list:
                try:
                    with open(pathway + post_type + file + '/tweets/%s' %tweet_file) as f: # read json files
                        data_dict = json.load(f)
                except:
                    continue
                data = data_dict['user']
                    
                uid.append(data['id_str'])
                num_friends = str(data['friends_count'])
                # skip num word description
                # skip num word name
                num_followers = str(data['followers_count'])
                num_statuses = str(data['statuses_count'])
                verified = str(data['verified']-0)
                geo_position = str(data['geo_enabled']-0)
                # skip time
                num_favorite = str(data['favourites_count'])
                profile_background = str(data['profile_use_background_image']-0)
                profile = str(data['default_profile']-0)
                profile_image = str(data['default_profile_image']-0)

                user_features.append([num_friends, num_followers, num_statuses, 
                                     verified, geo_position, num_favorite, profile_background,
                                     profile, profile_image])

                
                
            if not os.path.exists(pathway + post_type + file + '/retweets'):
                continue;
            tweets_list = os.listdir(pathway + post_type + file + '/retweets')
            for tweet_file in tweets_list:
                try:
                    with open(pathway + post_type + file + '/retweets/%s' %tweet_file) as f: # read json files
                        data_dict = json.load(f)
                except:
                    continue

                for datas in data_dict['retweets']:
                    data = datas['user']
                    uid.append(data['id_str'])
                    num_friends = str(data['friends_count'])
                    # skip num word description
                    # skip num word name
                    num_followers = str(data['followers_count'])
                    num_statuses = str(data['statuses_count'])
                    verified = str(data['verified']-0)
                    geo_position = str(data['geo_enabled']-0)
                    # skip time
                    num_favorite = str(data['favourites_count'])
                    profile_background = str(data['profile_use_background_image']-0)
                    profile = str(data['default_profile']-0)
                    profile_image = str(data['default_profile_image']-0)

                    user_features.append([num_friends, num_followers, num_statuses, 
                                         verified, geo_position, num_favorite, profile_background,
                                         profile, profile_image])

    
with open(output_dir + 'FNN_user_features.txt', 'w') as f:
    for i in tqdm(range(len(uid)), 'writing user features'):
        f.write('u%s: ' %uid[i])
        f.write(' '.join(user_features[i]))
        f.write('\n')
    
    
    

