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

    data_path = '/FakeNewsNet/' # fake news net page
    combo_path = '/FakeNewsNet/FNN_input/%s/%s/' % (dataset, folder_name) # store 5n5p100u input data 
    roberta_path = '/FakeNewsNet/text_embeddings/user_description/' # get user roberta embedding here
    image_path = '/FakeNewsNet/visual_features/'
    post_path = 'FNN_input/%s/%s/normalized_post_nodes/' % (dataset, folder_name) # store post nodes here 
    user_path = 'FNN_input/%s/%s/normalized_user_nodes/' % (dataset, folder_name) # store user nodes here
    news_path = 'FNN_input/%s/%s/normalized_news_nodes/' % (dataset, folder_name) # store news nodes here

    neighbors_path = '/rwr_results/%s/' %folder_name # read neighbor list from here


    if not os.path.exists(data_path + user_path):
        os.makedirs(data_path + user_path)


    # In[25]:


    print("load user neighbors.txt")
    with open(neighbors_path + 'u_neighbors.txt', 'r') as f:
        user_neighbors = f.readlines()


    # In[26]:


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


    # In[29]:


    # user labels


    # In[33]:


    # user features
    print('get all user features...')
    f = open(data_path + 'FNN_user_features_onehot.txt', 'r')
    user_f = f.readlines()
    f.close()
    user_feature = dict()
    for line in tqdm(user_f, desc = 'get all user features'):
        line = line.split(' ', 1)
        user_feature[line[0][1:-1]] = line[1]


    # In[32]:


    # user description
    if os.path.exists('user_description.txt'):
        print("load user description from txt file..")
        with open('user_description.txt', 'r') as f:
            user_d = np.loadtxt(f).astype(float)
    else:
        user_d = dict()
        count = 0
        unincluded_user = []
        for user in tqdm(user_id, desc='get all user description'):
            try:
                f = open(roberta_path + user + '.txt', 'r')
                description = np.loadtxt(f, delimiter = ' ')
                f.close()
                if len(description) == 0:
                    continue
                user_d[user] = description
            except:
                unincluded_user.append('u' + user)
                # print("%s not found." %user)
                count += 1
                pass
        print("%d/%d users not found" %(count, len(user_id)))
        # with open(combo_path + 'user_description.txt', 'w') as f:
            # np.savetxt(f, user_d)
        with open(combo_path + 'unincluded_users.txt', 'w') as f:
            f.write(' '.join(unincluded_user))


    # In[ ]:


    print('normalize user description...')
    scaler = preprocessing.StandardScaler().fit(list(user_d.values()))
    normalized_d = scaler.transform(list(user_d.values()))
    keys = list(user_d.keys())
    for i in range(len(keys)):
        user_d[keys[i]] = list(map(str, normalized_d[i]))

    padding_d = ['0'] * len(normalized_d[0])



    print("%d users." %len(user_id))
    print("%d user_feature" %len(user_feature))
    print("%d user description" %len(normalized_d))
    print("%d news neighbors and %d post neighbors and %d user neighbors" %(len(all_post_neighbors), 
                                                                            len(all_user_neighbors),
                                                                           len(all_news_neighbors)))
    problem_users = []
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
                    problem_users.append(user_id[i])
                try:
                    f.write(' '.join(user_d[user_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_d) + '\n')
                    problem_users.append(user_id[i])

                f.write(' '.join(all_news_neighbors[i]) + '\n')
                f.write(' '.join(all_post_neighbors[i]) + '\n')
                f.write(' '.join(all_user_neighbors[i]) + '\n')


    with open(combo_path + 'problem_users.txt', 'w') as f:
        f.write(' '.join(problem_users))

    # In[2]:





