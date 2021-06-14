#!/usr/bin/env python
# coding: utf-8


import os
from tqdm import tqdm
import json

data_path = 'data/PHEME/'
output_dir = 'data/processed_data/PHEME/'

event_list = ['charliehebdo-all-rnr-threads/', 'ebola-essien-all-rnr-threads/', 'ferguson-all-rnr-threads/',
          'germanwings-crash-all-rnr-threads/', 'prince-toronto-all-rnr-threads/', 'gurlitt-all-rnr-threads/',
          'putinmissing-all-rnr-threads/', 'ottawashooting-all-rnr-threads/', 'sydneysiege-all-rnr-threads/']

user_features = list()

for event in tqdm(event_list):
    for news_type in ['rumours/', 'non-rumours/']:

        event_path = data_path + event + news_type
        post_list = os.listdir(event_path)
        try:
            post_list.remove('.DS_Store')
        except:
            pass
        for post in tqdm(post_list):
            for post_type in ['/source-tweets/', '/reactions/']:
                tweet_path = event_path + post + post_type
                tweet_list = os.listdir(tweet_path)
                try:
                    tweet_list.remove('.DS_Store')
                except:
                    pass
                for tweet in tweet_list:
                    with open(tweet_path + tweet) as json_file:
                        tweet_dict = json.load(json_file)
                    user_info = tweet_dict['user']
                    user_id = user_info['id_str']
                    verified = str(user_info['verified'] - 0)
                    statuses_count = str(user_info['statuses_count'])
                    followers_count = str(user_info['followers_count'])
                    friends_count = str(user_info['friends_count'])
                    geo_enabled = str(user_info['geo_enabled'] - 0)
                    favorites_count = str(user_info['favourites_count'])
                    
                    # favorite_count = str(news_dict['favorite_count'])
                    # followers_count = str(news_dict['followers_count'])
                    user_features.append([user_id, verified, statuses_count, followers_count, 
                                          favorites_count, friends_count, geo_enabled])

if not os.exists.path(output_dir):
    os.mkdir(output_dir)
    
with open(output_dir + 'user_features_pheme.txt', 'w') as f:
    for [user_id, verified, statuses_count, followers_count, 
          favorites_count, friends_count, geo_enabled] in user_features:
        f.write('%s: %s %s %s %s %s %s\n' %(user_id, verified, statuses_count, followers_count, 
                                              favorites_count, friends_count, geo_enabled,))
