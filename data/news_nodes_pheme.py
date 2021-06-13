#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing

def create_news_nodes(folder_name):
    
    news_text_path = '/pheme-figshare/text_embeddings/tweet_text/' # get user roberta embedding here
    user_text_path = '/pheme-figshare/text_embeddings/user_description/'
    
    data_path = '/pheme-figshare/' # fake news net page
    combo_path = '/pheme-figshare/pheme_input/%s/' % folder_name # store 5n5p100u input data 
    post_path = 'pheme_input/%s/normalized_post_nodes/' % folder_name # store post nodes here 
    user_path = 'pheme_input/%s/normalized_user_nodes/' % folder_name # store user nodes here
    news_path = 'pheme_input/%s/normalized_news_nodes/' % folder_name# store news nodes here

    neighbors_path = '/rwr_results/%s/' % folder_name # read neighbor list from here




    if not os.path.exists(data_path + news_path):
        os.makedirs(data_path + news_path)

    print('get news labels...')
    with open (data_path + 'news_label_pheme.txt', 'r') as f:
        data = f.readlines()
    news_label = dict()
    for line in data:
        line = line.rstrip('\n').split(': ')
        news_label[line[0]] = line[1] 
    # In[25]:

    print("load news neighbors.txt")
    with open(neighbors_path + 'n_neighbors.txt', 'r') as f:
        news_neighbors = f.readlines()

    # In[26]:


    print('get all neighbors...')
    all_post_neighbors = []
    all_user_neighbors = []
    all_news_neighbors = []
    news_id = []
    for news in tqdm(news_neighbors, desc='get all neighbors...'):
        news = news.split()
        news_id.append(news[0][1:-1])

        n_neighbors = []
        p_neighbors = []
        u_neighbors = []
        for neighbor in news[1:]:
            if neighbor[0] == 'p':
                p_neighbors.append(neighbor[1:])
            elif neighbor[0] == 'u':
                u_neighbors.append(neighbor[1:])
            elif neighbor[0] == 'n':
                n_neighbors.append(neighbor[1:])
        all_post_neighbors.append(p_neighbors)
        all_user_neighbors.append(u_neighbors)
        all_news_neighbors.append(n_neighbors)



    # In[32]:


    # news text


    news_content = dict()

    content_count = 0

    unincluded_news_content = list()
    for news in tqdm(news_id, desc='get all news content'):

        try:
            f = open(news_text_path + news + '.txt', 'r')
            content = np.loadtxt(f, delimiter = ' ')
            f.close()
            news_content[news] = content
        except:
            unincluded_news_content.append('n' + news)
            print("%s not found." %news)
            content_count += 1
            pass

    print("%d news content not found" %content_count)


    with open(combo_path + 'unincluded_news_content.txt', 'w') as f:
        f.write(' '.join(unincluded_news_content))


    # In[ ]:


    print('normalize news content...')   
    scaler = preprocessing.StandardScaler().fit(list(news_content.values()))
    normalized_content = scaler.transform(list(news_content.values()))
    keys = list(news_content.keys())
    for i in range(len(keys)):
        news_content[keys[i]] = list(map(str, normalized_content[i]))


    padding_content = ['0'] * len(normalized_content[0])


    print("%d news." %len(news_id))
    print("%d news content" %len(normalized_content))
    print("%d news neighbors and %d post neighbors and %d user neighbors" %(len(all_post_neighbors), 
                                                                            len(all_user_neighbors),
                                                                           len(all_news_neighbors)))

    for batch in tqdm(range(len(news_id)//5000 + 1), desc='writing batches......'):
        with open(data_path + news_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing news nodes.....'):
                if (i >= len(news_id)):
                    break
                f.write('n ' + news_id[i] + ' %s' %news_label[news_id[i]] + '\n')

                try:
                    f.write(' '.join(news_content[news_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_content) + '\n')


                f.write(' '.join(all_news_neighbors[i]) + '\n')
                f.write(' '.join(all_post_neighbors[i]) + '\n')
                f.write(' '.join(all_user_neighbors[i]) + '\n')
