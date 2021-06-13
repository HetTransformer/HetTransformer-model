#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
from tqdm import tqdm

data_path = ''

event_list = ['charliehebdo-all-rnr-threads/', 'ebola-essien-all-rnr-threads/', 'ferguson-all-rnr-threads/',
          'germanwings-crash-all-rnr-threads/', 'prince-toronto-all-rnr-threads/', 'gurlitt-all-rnr-threads/',
          'putinmissing-all-rnr-threads/', 'ottawashooting-all-rnr-threads/', 'sydneysiege-all-rnr-threads/']

news_labels = list()

for event in tqdm(event_list):
    for news_type in ['rumours/', 'non-rumours/']:
        if news_type == 'rumours/':
            label = '0'
        else:
            label = '1'
        event_path = data_path + event + news_type
        post_list = os.listdir(event_path)
        try:
            post_list.remove('.DS_Store')
        except:
            pass
        for post in tqdm(post_list):
            news_path = event_path + post + '/source-tweets/'
            news_list = os.listdir(news_path)
            try:
                news_list.remove('.DS_Store')
            except:
                pass
            news = news_list[0].rstrip('.json')
            news_labels.append([news, label])

with open(data_path + 'news_label.txt', 'w') as f:
    for news in news_labels:
        f.write(': '.join(news) + '\n')
