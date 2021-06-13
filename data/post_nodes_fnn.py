#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing
from multiprocessing import Pool, Manager


def create_FNN_post_nodes(dataset, folder_name):
    
    data_path = '/FakeNewsNet/' # fake news net page
    combo_path = '/FakeNewsNet/FNN_input/%s/%s/' % (dataset, folder_name) # store 5n5p100u input data 
    image_path = '/FakeNewsNet/visual_features/'
    post_path = 'FNN_input/%s/%s/normalized_post_nodes/' % (dataset, folder_name) # store post nodes here 
    user_path = 'FNN_input/%s/%s/normalized_user_nodes/' % (dataset, folder_name) # store user nodes here
    news_path = 'FNN_input/%s/%s/normalized_news_nodes/' % (dataset, folder_name) # store news nodes here
    roberta_path = '/FakeNewsNet/text_embeddings/tweet_text/' # get user roberta embedding here

    neighbors_path = '/rwr_results/%s/' %folder_name # read neighbor list from here


    if not os.path.exists(data_path + post_path):
        os.makedirs(data_path + post_path)

    # In[25]:


    print("load post neighbors.txt")
    with open(neighbors_path + 'p_neighbors.txt', 'r') as f:
        post_neighbors = f.readlines()


    # In[26]:


    print('get all neighbors...')
    all_post_neighbors = []
    all_user_neighbors = []
    all_news_neighbors = []
    post_id = []
    for post in tqdm(post_neighbors, desc='get all neighbors...'):
        post = post.split()
        post_id.append(post[0][1:-1])

        n_neighbors = []
        p_neighbors = []
        u_neighbors = []
        for neighbor in post[1:]:
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


    # In[33]:


    # post features
    print('get all post features...')
    f = open(data_path + 'FNN_post_features.txt', 'r')
    post_f = f.readlines()
    f.close()
    post_feature = dict()
    for line in tqdm(post_f, desc = 'get all post features'):
        line = line.split(' ', 1)
        post_feature[line[0][1:-1]] = line[1]

    # padding features
    int_features = list()
    for i in post_feature.keys():
        int_features.append(list(map(float, post_feature[i].split())))
    padding_features = np.mean(int_features, 0)
    padding_features = list(map(str, padding_features))


    # In[32]:


    # post description
    if os.path.exists('post_content.txt'):
        print("load post content from txt file..")
        with open('post_content.txt', 'r') as f:
            post_content = np.loadtxt(f).astype(float)
    else:
        post_content = dict()
        count = 0
        unincluded_post = []
        for post in tqdm(post_id, desc='get all post content'):
            try:
                f = open(roberta_path + post + '.txt', 'r')
                description = np.loadtxt(f, delimiter = ' ')
                f.close()
                post_content[post] = description
            except:
                unincluded_post.append('p' + post)
                # print("%s not found." %post)
                count += 1
                pass
        print("%d/%d post content not found" %(count, len(post_id)))
        # with open(combo_path + 'post_description.txt', 'w') as f:
            # np.savetxt(f, post_content)
        with open(combo_path + 'unincluded_posts.txt', 'w') as f:
            f.write(' '.join(unincluded_post))



    # In[ ]:


    print('normalize post content...')
    scaler = preprocessing.StandardScaler().fit(list(post_content.values()))
    normalized_content = scaler.transform(list(post_content.values()))
    keys = list(post_content.keys())
    for i in range(len(keys)):
        post_content[keys[i]] = list(map(str, normalized_content[i]))

    # padding description
    padding_description = ['0'] * len(normalized_content[0])

    # In[ ]:



    print("%d posts." %len(post_id))
    print("%d post_features" %len(post_feature))
    print("%d post content" %len(normalized_content))
    print("%d news neighbors and %d post neighbors and %d user neighbors" %(len(all_post_neighbors), 
                                                                            len(all_user_neighbors),
                                                                           len(all_news_neighbors)))

    problem_posts = list()
    for batch in tqdm(range(len(post_id)//5000 + 1), desc='writing batches......'):
        with open(data_path + post_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing post nodes.....'):
                if (i >= len(post_id)):
                    break
                f.write('p ' + post_id[i] + '\n')
                try:
                    f.write(post_feature[post_id[i]])
                except:
                    f.write(' '.join(padding_features) + '\n')
                    problem_posts.append(post_id[i])
                try:
                    f.write(' '.join(post_content[post_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_description) + '\n')
                    problem_posts.append(post_id[i])
                f.write(' '.join(all_news_neighbors[i]) + '\n')
                f.write(' '.join(all_post_neighbors[i]) + '\n')
                f.write(' '.join(all_user_neighbors[i]) + '\n')


    with open(combo_path + 'problem_posts.txt', 'w') as f:
        f.write(' '.join(problem_posts))


