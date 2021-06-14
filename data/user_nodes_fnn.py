#!/usr/bin/env python
# coding: utf-8

# In[2]:

# import 


import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing

from multiprocessing import Pool, Manager
# decide how many cpus you need to load with.

def create_FNN_user_nodes(dataset, folder_name):

    data_path = 'data/processed_data/FakeNewsNet/%s/' % dataset # fake news net page
    combo_path = 'data/processed_data/FakeNewsNet/%s/%s/' % (dataset, folder_name) # store 5n5p100u input data 
    roberta_path = 'data/processed_data/FakeNewsNet/text_embeddings/user_description/' # get user roberta embedding here
    image_path = 'data/processed_data/FakeNewsNet/%s/visual_features/' % dataset
    post_path = '%s/normalized_post_nodes/' % (dataset, folder_name) # store post nodes here 
    user_path = '%s/normalized_user_nodes/' % (dataset, folder_name) # store user nodes here
    news_path = '%s/normalized_news_nodes/' % (dataset, folder_name) # store news nodes here

    neighbors_path = 'data/rwr_results/%s/' %folder_name # read neighbor list from here


    if not os.path.exists(data_path + user_path):
        os.makedirs(data_path + user_path)


    print("load user neighbors.txt")
    with open(neighbors_path + 'u_neighbors.txt', 'r') as f:
        user_neighbors = f.readlines()


    print('get all neighbors...')
    all_post_neighbors = []
    all_user_neighbors = []
    all_news_neighbors = []
    user_id = []
    for user in tqdm(user_neighbors, desc='get all neighbors...'):
        user = user.split()
        user_id.append(user[0][1:-1])

        n_neighbors = []
        p_neighbors = []
        u_neighbors = []
        for neighbor in user[1:]:
            if neighbor[0] == 'p':
                p_neighbors.append(neighbor[1:])
            elif neighbor[0] == 'u':
                u_neighbors.append(neighbor[1:])
            elif neighbor[0] == 'n':
                n_neighbors.append(neighbor[1:])
        all_post_neighbors.append(p_neighbors)
        all_user_neighbors.append(u_neighbors)
        all_news_neighbors.append(n_neighbors)



    # user features
    print('get all user features...')
    f = open(data_path + 'user_features_onehot.txt', 'r')
    
    user_f = f.readlines()
    f.close()
    user_feature = dict()
    for line in tqdm(user_f, desc = 'get all user features'):
        line = line.split(' ', 1)
        user_feature[line[0][1:-1]] = line[1]

    # user description
    if os.path.exists('user_description.txt'):
        print("load user description from txt file..")
        with open('user_description.txt', 'r') as f:
            user_d = np.loadtxt(f).astype(float)
    else:
        user_d = dict()
    
        for user in tqdm(user_id, desc='get all user description'):
            try:
                f = open(roberta_path + user + '.txt', 'r')
                description = np.loadtxt(f, delimiter = ' ')
                f.close()
                if len(description) == 0:
                    continue
                user_d[user] = description
            except:
                pass

    print('normalize user description...')
    scaler = preprocessing.StandardScaler().fit(list(user_d.values()))
    normalized_d = scaler.transform(list(user_d.values()))
    keys = list(user_d.keys())
    for i in range(len(keys)):
        user_d[keys[i]] = list(map(str, normalized_d[i]))

    padding_d = ['0'] * len(normalized_d[0])

    for batch in tqdm(range(len(user_id)//5000 + 1), desc='writing batches......'):
        with open(data_path + user_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing user nodes.....'):
                if (i >= len(user_id)):
                    break
                f.write('u ' + user_id[i] + '\n')
                try:
                    f.write(user_feature[user_id[i]].strip('\n') + '\n')
                except:
                    f.write(user_feature['padding'].strip('\n') + '\n')
                try:
                    f.write(' '.join(user_d[user_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_d) + '\n')

                f.write(' '.join(all_news_neighbors[i]) + '\n')
                f.write(' '.join(all_post_neighbors[i]) + '\n')
                f.write(' '.join(all_user_neighbors[i]) + '\n')




