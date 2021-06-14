#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing

def create_FNN_news_nodes(dataset, folder_name):
    
    roberta_title_path = 'data/processed_data/text_embeddings/news_title/' # get user roberta embedding here
    roberta_content_path = 'data/processed_data/FakeNewsNet/text_embeddings/news_text/'
    data_path = 'data/processed_data/FakeNewsNet/%s/' % dataset  # fake news net page
    combo_path = 'data/processed_data/FakeNewsNet/%s/%s/' % (dataset, folder_name) # store 5n5p100u input data 
    image_path = 'data/processed_data/FakeNewsNet/%s/visual_features/' % dataset
    post_path = 'data/processed_data/FakeNewsNet/%s/%s/normalized_post_nodes/' % (dataset, folder_name) # store post nodes here 
    user_path = 'data/processed_data/FakeNewsNet/%s/%s/normalized_user_nodes/' % (dataset, folder_name) # store user nodes here
    news_path = 'data/processed_data/FakeNewsNet/%s/%s/normalized_news_nodes/' % (dataset, folder_name) # store news nodes here
    roberta_path = 'data/processed_data/FakeNewsNet/text_embeddings/tweet_text/' # get user roberta embedding here

    neighbors_path = 'data/rwr_results/%s/' %folder_name # read neighbor list from here



    if not os.path.exists(news_path):
        os.makedirs(news_path)


    print('get news labels...')
    with open (data_path + 'news_label.txt', 'r') as f:
        data = f.readlines()
    news_label = dict()
    for line in data:
        line = line.split()
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


    # news title
    if os.path.exists('news_title.txt'):
        print("load news title from txt file..")
        with open('news_title.txt', 'r') as f:
            news_title = np.loadtxt(f).astype(float)
    if os.path.exists('new_content.txt'):
        print("load new content from txt file..")
        with open('news_content.txt', 'r') as f:
            news_content = np.loadtxt(f).astype(float)
    if os.path.exists('new_image.txt'):
        print("load new image from txt file..")
        with open('news_image.txt', 'r') as f:
            news_image = np.loadtxt(f).astype(float)
    else:
        news_title = dict()
        news_content = dict()
        news_image = dict()


        with open(image_path + 'white.txt', 'r') as f:
            padding_image = np.loadtxt(f)
        print("white img is of size %d" %len(padding_image))
        news_image['PADDING'] = padding_image

        for news in tqdm(news_id, desc='get all news title and content and image'):
            try:
                f = open(roberta_title_path + news + '.txt', 'r')
                title = np.loadtxt(f, delimiter = ' ')
                f.close()
                news_title[news] = title
            except:
                pass
            try:
                f = open(roberta_content_path + news + '.txt', 'r')
                content = np.loadtxt(f, delimiter = ' ')
                f.close()
                news_content[news] = content
            except:  
                pass
            try:
                f = open(image_path + news + '.txt', 'r')
                image = np.loadtxt(f)
                f.close()
                news_image[news] = image
            except:
                pass

    print('normalize news title and content and image...')
    scaler = preprocessing.StandardScaler().fit(list(news_title.values()))
    normalized_title = scaler.transform(list(news_title.values()))
    keys = list(news_title.keys())
    for i in range(len(keys)):
        news_title[keys[i]] = list(map(str, normalized_title[i]))


    scaler = preprocessing.StandardScaler().fit(list(news_content.values()))
    normalized_content = scaler.transform(list(news_content.values()))
    keys = list(news_content.keys())
    for i in range(len(keys)):
        news_content[keys[i]] = list(map(str, normalized_content[i]))


    scaler = preprocessing.StandardScaler().fit(list(news_image.values()))
    normalized_image = scaler.transform(list(news_image.values()))
    keys = list(news_image.keys())
    for i in range(len(keys)):
        news_image[keys[i]] = list(map(str, normalized_image[i]))

    padding_title = ['0'] * len(normalized_title[0])
    padding_content = ['0'] * len(normalized_content[0])
    padding_image = news_image['PADDING']

    for batch in tqdm(range(len(news_id)//5000 + 1), desc='writing batches......'):
        with open(news_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing news nodes.....'):
                if (i >= len(news_id)):
                    break
                f.write('n ' + news_id[i] + ' %s' %news_label[news_id[i]] + '\n')
                try:
                    f.write(' '.join(news_title[news_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_title) + '\n')
                try:
                    f.write(' '.join(news_content[news_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_content) + '\n')
                try:
                    f.write(' '.join(news_image[news_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_image) + '\n')

                f.write(' '.join(all_news_neighbors[i]) + '\n')
                f.write(' '.join(all_post_neighbors[i]) + '\n')
                f.write(' '.join(all_user_neighbors[i]) + '\n')

