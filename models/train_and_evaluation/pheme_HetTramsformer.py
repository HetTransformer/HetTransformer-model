#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
#import torch.nn.init as init
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from os import path
from torch.autograd import Variable
import math
from torch_position_embedding import PositionEmbedding
import random

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# news-news; news-post; post-user; user-user


class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_news=[], neighbor_list_post=[], neighbor_list_user=[],
                 label=None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label  # only post node, user node = default = None
        self.neighbors_news = neighbor_list_news  # [(id)]
        self.neighbors_post = neighbor_list_post
        self.neighbors_user = neighbor_list_user

def neighbor_loader(pathway):
    neighbor_dict_post = {}
    neighbor_dict_user = {}
    neighbor_dict_news = {}
    neighbor_dict_n_p_u = {}
    neighbor_dict_n = {}
    f = open(pathway)
    Lines = f.readlines()
    for i in range(len(Lines)):
        neighbor_list = Lines[i].split()
        neighbor_dict_n_p_u[neighbor_list[0][1:-1]] = [(item[0], item[1:]) for item in neighbor_list[1:]]
        neighbor_dict_n[neighbor_list[0][1:-1]] = [(item[0], item[1:]) for item in neighbor_list[1:] if item[0] == 'n']
        neighbor_dict_news[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'n']
        neighbor_dict_user[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'u']
        neighbor_dict_post[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'p']
    return neighbor_dict_n_p_u, neighbor_dict_n, neighbor_dict_news, neighbor_dict_post, neighbor_dict_user

neighbor_dict = neighbor_loader('data/processed_data/PHEME/pheme_n_neighbors.txt')

def data_loader(pathway='data/processed_data/PHEME/normalized_news_nodes/', node_type="news"):
    if node_type == "news":
        news_node = []
        news_id = []
        news_label = []
        news_embed = []
        news_p_neigh = []
        news_n_neigh = []
        news_u_neigh = []
        for i in range(2):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            
            for j in range(len(Lines)):               
                if j % 3 == 0:
                    _, id_, label = Lines[j].split()
                    news_id.append(id_)
                    news_label.append(int(label))
                    embed = []
                if j % 3 == 1:
                    embed.append(list(map(float, Lines[j].split())))
                if j % 3 == 1:
                    news_embed.append(embed)
                if j % 3 == 2:
                    news_n_neigh.append([])
                    news_p_neigh.append([])
                    news_u_neigh.append([])
            f.close()
        for i in range(len(news_id)):
            node = Het_Node(node_type="news", node_id=news_id[i], embed=news_embed[i],
                            neighbor_list_news=news_n_neigh[i], neighbor_list_post=news_p_neigh[i], neighbor_list_user=news_u_neigh[i], label=news_label[i])
            news_node.append(node)
        #padding_node = Het_Node(node_type="news", node_id=padding_id, embed=padding_embed,
        #                    neighbor_list_news=padding_news_n_neigh, neighbor_list_post=padding_news_p_neigh, neighbor_list_user=padding_news_u_neigh)
        return news_node #, padding_node

    elif node_type == 'post':
        post_node = []
        post_id = []
        post_embed = []
        for i in range(18):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):              
                if j % 3 == 0:
                    _, id_ = Lines[j].split()
                    post_id.append(id_)
                    embed = []
                if j % 3 == 1:
                    embed.append(list(map(float, Lines[j].split())))
                if j % 3 == 1:
                    post_embed.append(embed)
            f.close()
        for i in range(len(post_id)):
            node = Het_Node(node_type="post", node_id=post_id[i], embed=post_embed[i])
            post_node.append(node)
        # padding_node = Het_Node(node_type='post', node_id=padding_id, embed=padding_embed)
        return post_node #, padding_node

    else:
        user_node = []
        user_id = []
        user_embed = []
        for i in range(11):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j % 4 == 0:
                    id_ = Lines[j].split()
                    user_id.append(id_[1])
                    embed = []
                if j % 4 == 1 or j % 4 == 2:
                    embed.append(list(map(float, Lines[j].split())))
                if j % 4 == 2:
                    user_embed.append(embed)
            f.close()
        for i in range(len(user_id)):
            node = Het_Node(node_type="user", node_id=user_id[i], embed=user_embed[i])
            user_node.append(node)
        # padding_node = Het_Node(node_type='user', node_id=padding_id, embed=padding_embed)
        return user_node #, padding_node
    
post_nodes = data_loader(pathway='data/processed_data/PHEME/batch/normalized_post_nodes/', node_type="post")
news_nodes = data_loader(pathway='data/processed_data/PHEME/batch/normalized_news_nodes/', node_type="news")
user_nodes = data_loader(pathway='data/processed_data/PHEME/batch/normalized_user_nodes/', node_type="user")
news_emb_dict = {}
post_emb_dict = {}
user_emb_dict = {}

for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb
for news in news_nodes:
    news_emb_dict[news.node_id] = news.emb


# In[2]:


news_nodes_real = []
news_nodes_fake = []
for node in news_nodes:
    if node.label == 1:
        news_nodes_real.append(node)
    else:
        news_nodes_fake.append(node)
ratio = len(news_nodes_real) / len(news_nodes_fake)


# In[3]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class HetTransformer(nn.Module):
    # features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim,
                 n_hidden_dim, n_ini_hidden_dim, n_output_dim,
                 u_hidden_dim, u_ini_hidden_dim, u_output_dim,
                 p_hidden_dim, p_ini_hidden_dim, p_output_dim,
                 out_embed_d,
                 outemb_d=1, n_num_layers=1, u_num_layers=1, p_num_layers=1, num_layers=1, n_batch_size=1,
                 u_batch_size=1, p_batch_size=1, content_dict={}, n_rnn_type='LSTM', u_rnn_type='LSTM',
                 p_rnn_type='LSTM', rnn_type='LSTM', embed_d=200, symmetry=False, GCN_out2=300, GCN_out1=600, 
                 GCN_in=200, use_bias=True, attn_heads = 2, d_model = 512, 
                 self_attn_heads = 1, enc_layers=3, dec_layers=3):
        super(HetTransformer, self).__init__()
        self.input_dim = input_dim
        self.ini_hidden_dim = ini_hidden_dim
        self.hidden_dim = out_embed_d
        self.num_layers = num_layers
        self.embed_d = out_embed_d
        self.n_input_dim = out_embed_d
        self.n_hidden_dim = n_hidden_dim
        self.n_ini_hidden_dim = n_ini_hidden_dim
        self.n_batch_size = n_batch_size
        self.n_output_dim = out_embed_d
        self.n_num_layers = n_num_layers
        self.n_rnn_type = n_rnn_type
        self.u_input_dim = out_embed_d
        self.u_hidden_dim = u_hidden_dim
        self.u_ini_hidden_dim = u_ini_hidden_dim
        self.u_batch_size = u_batch_size
        self.u_output_dim = out_embed_d
        self.u_num_layers = u_num_layers
        self.u_rnn_type = u_rnn_type
        self.p_input_dim = out_embed_d
        self.p_hidden_dim = p_hidden_dim
        self.p_ini_hidden_dim = p_ini_hidden_dim
        self.p_batch_size = p_batch_size
        self.p_output_dim = out_embed_d
        self.p_num_layers = p_num_layers
        self.p_rnn_type = p_rnn_type
        self.out_embed_d = out_embed_d
        self.outemb_d = outemb_d
        self.content_dict = content_dict
        self.d_model = d_model
        
        self.GCN_in = out_embed_d
        self.GCN_out1 = GCN_out1
        self.GCN_out2 = out_embed_d * 2
        self.use_bias = use_bias
        self.symmetry = symmetry
        self.GCN_W1_user = torch.nn.Parameter(torch.FloatTensor(self.GCN_in, self.GCN_out1), requires_grad=True)
        self.GCN_W2_user = torch.nn.Parameter(torch.FloatTensor(self.GCN_out1, self.GCN_out2), requires_grad=True)
        self.GCN_W1_post = torch.nn.Parameter(torch.FloatTensor(self.GCN_in, self.GCN_out1), requires_grad=True)
        self.GCN_W2_post = torch.nn.Parameter(torch.FloatTensor(self.GCN_out1, self.GCN_out2), requires_grad=True)
        self.GCN_W1_news = torch.nn.Parameter(torch.FloatTensor(self.GCN_in, self.GCN_out1), requires_grad=True)
        self.GCN_W2_news = torch.nn.Parameter(torch.FloatTensor(self.GCN_out1, self.GCN_out2), requires_grad=True)
        if self.use_bias:
            self.user_bias1 = nn.Parameter(torch.Tensor(self.GCN_out1), requires_grad=True)
            self.post_bias1 = nn.Parameter(torch.Tensor(self.GCN_out1), requires_grad=True)
            self.news_bias1 = nn.Parameter(torch.Tensor(self.GCN_out1), requires_grad=True)
            self.user_bias2 = nn.Parameter(torch.Tensor(self.GCN_out2), requires_grad=True)
            self.post_bias2 = nn.Parameter(torch.Tensor(self.GCN_out2), requires_grad=True)
            self.news_bias2 = nn.Parameter(torch.Tensor(self.GCN_out2), requires_grad=True)
        else:
            self.register_parameter('bias', None)
            
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=201)
        self.pos_decoder = PositionalEncoding(d_model, dropout=0.1, max_len=201)
        self.type_encoder = nn.Embedding(3, d_model, padding_idx=0)
        self.n_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.transformer = nn.Transformer(d_model=d_model, nhead=attn_heads, num_encoder_layers=dec_layers, num_decoder_layers=enc_layers, dim_feedforward=512, 
                                          dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=attn_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=enc_layers)
        self.init_linear_text = nn.Linear(self.input_dim[0], self.hidden_dim)
        #self.init_linear_text_post = nn.Linear(self.input_dim[0], self.hidden_dim)
        #self.init_linear_text_user = nn.Linear(self.input_dim[0], self.hidden_dim)
        self.init_linear_feature = nn.Linear(self.input_dim[1], self.hidden_dim)
        # Define the LSTM layer
        #self.news_content_LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
        #                                                     batch_first=True,
        #                                                     bidirectional=True)
        self.news_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.4)
        #self.post_content_LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
        #                                                     batch_first=True,
        #                                                     bidirectional=True)
        self.post_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.4)
        #self.user_content_LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
        #                                                     batch_first=True,
        #                                                     bidirectional=True)
        self.user_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.4)
        #self.LSTM_other_user = eval('nn.' + rnn_type)(self.ini_hidden_dim[1], self.hidden_dim, self.num_layers,
        #                                              batch_first=True,
        #                                              bidirectional=True)
        self.attention_other_user = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.4)
        self.layernorm1 = nn.LayerNorm([1, out_embed_d])
        self.layernorm2 = nn.LayerNorm([1, out_embed_d])
        self.layernorm3 = nn.LayerNorm([1, out_embed_d])
        self.layernorm4 = nn.LayerNorm([1, out_embed_d])
        self.n_init_linear = nn.Linear(self.n_input_dim, self.n_hidden_dim)
        # self.u_init_dropout = nn.Dropout(p=0.2)
        # self.n_LSTM = eval('nn.' + self.n_rnn_type)(self.n_ini_hidden_dim, self.n_hidden_dim, self.n_num_layers,
        #                                            batch_first=True, bidirectional=True)
        self.n_attention = nn.MultiheadAttention(self.n_hidden_dim, self_attn_heads, dropout = 0.4)
        self.n_linear = nn.Linear(self.n_hidden_dim, self.n_output_dim)
        self.n_dropout = nn.Dropout(p=0.5)
        self.u_init_linear = nn.Linear(self.u_input_dim, self.u_hidden_dim)
        # self.u_init_dropout = nn.Dropout(p=0.2)
        #self.u_LSTM = eval('nn.' + self.u_rnn_type)(self.u_ini_hidden_dim, self.u_hidden_dim, self.u_num_layers,
        #                                            batch_first=True, bidirectional=True)
        self.u_attention = nn.MultiheadAttention(self.u_hidden_dim, self_attn_heads, dropout = 0.4)
        self.u_linear = nn.Linear(self.u_hidden_dim, self.u_output_dim)
        self.u_dropout = nn.Dropout(p=0.5)
        self.p_init_linear = nn.Linear(self.p_input_dim, self.p_hidden_dim)
        # self.p_init_dropout = nn.Dropout(p=0.2)
        #self.p_LSTM = eval('nn.' + self.p_rnn_type)(self.p_ini_hidden_dim, self.p_hidden_dim, self.p_num_layers,
        #                                            batch_first=True, bidirectional=True)
        self.p_attention = nn.MultiheadAttention(self.p_hidden_dim, self_attn_heads, dropout = 0.4)
        self.p_linear = nn.Linear(self.p_hidden_dim, self.p_output_dim)
        self.p_dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.out_dropout = nn.Dropout(p=0.2)
        self.out_linear = nn.Linear(self.out_embed_d, self.outemb_d)
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
 
    def Bi_RNN(self, neighbor_id, node_type, post_emb_dict, user_emb_dict, news_emb_dict):
        # Forward pass through initial hidden layer
        new_id = []
        if node_type == "news":
            input_content = []
            for i in neighbor_id:
                if ("news", i) not in self.content_dict:
                    input_content.append(news_emb_dict[i][0])
                    new_id.append(i)
            input_content = torch.Tensor(input_content)
            linear_input_content = self.init_linear_text(input_content)
            linear_input_content = linear_input_content.view(linear_input_content.shape[0], 1,
                                                             linear_input_content.shape[1])
            attention_out_content, self.hidden_text = self.news_content_attention_text(linear_input_content, linear_input_content, linear_input_content)
            attention_out_content = self.layernorm1(attention_out_content)
            concate = attention_out_content
            # concate = linear_input_content
        if node_type == "user":
            input_a = []
            input_b = []
            for i in neighbor_id:
                if ("user", i) not in self.content_dict:
                    input_a.append(user_emb_dict[i][0])
                    input_b.append(user_emb_dict[i][1])
                    new_id.append(i)
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_other = self.init_linear_feature(input_a)  # 22
            linear_input_content = self.init_linear_text(input_b)
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            linear_input_content = linear_input_content.view(linear_input_content.shape[0], 1, linear_input_content.shape[1])
            attention_out_other, self.hidden_other = self.attention_other_user(linear_input_other, linear_input_other, linear_input_other)
            attention_out_content, self.hidden_text = self.user_content_attention_text(linear_input_content, linear_input_content, linear_input_content)
            attention_out_other = self.layernorm2(attention_out_other)
            attention_out_content = self.layernorm3(attention_out_content)
            # concate = attention_out_content
            # concate = linear_input_content
            concate = torch.cat((attention_out_content, attention_out_other), 1)
        if node_type == "post":
            input_a = []
            for i in neighbor_id:
                if ("post", i) not in self.content_dict:
                    input_a.append(post_emb_dict[i][0])
                    new_id.append(i)
            input_a = torch.Tensor(input_a)
            linear_input_content = self.init_linear_text(input_a)
            linear_input_content = linear_input_content.view(linear_input_content.shape[0], 1,
                                                             linear_input_content.shape[1])
            attention_out_content, self.hidden_feature = self.post_content_attention_text(linear_input_content, linear_input_content, linear_input_content)
            attention_out_content = self.layernorm4(attention_out_content)
            concate = attention_out_content
            # concate = linear_input_content
        # mean pooling all the states
        mean_pooling = torch.mean(concate, 1)

        return mean_pooling

    # features: list of [(id)]
    def SameType_Agg_Bi_RNN(self, neighbor_id, node_type):
        content_embedings = self.Bi_RNN(neighbor_id, node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        aft_rnn_dict = {}
        for i in range(len(neighbor_id)):
            aft_rnn_dict[neighbor_id[i]] = content_embedings[i]
        if node_type == 'news':
            linear_input = self.n_init_linear(content_embedings)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            attention_out, hidden = self.n_attention(linear_input, linear_input, linear_input)
            last_state = self.n_linear(attention_out)
            last_state = self.n_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling, aft_rnn_dict
        if node_type == 'post':
            linear_input = self.p_init_linear(content_embedings)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            attention_out, hidden = self.p_attention(linear_input, linear_input, linear_input)
            last_state = self.p_linear(attention_out)
            last_state = self.p_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling, aft_rnn_dict
        if node_type == 'user':
            linear_input = self.u_init_linear(content_embedings)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            attention_out, hidden = self.u_attention(linear_input, linear_input, linear_input)
            last_state = self.u_linear(attention_out)
            last_state = self.u_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling, aft_rnn_dict

    def transformer_agg(self, het_node, neighbor_order_n_p_u, neighbor_order_n, npu=15, n=5):  # heterogeneous neighbor aggregation
        c_agg_batch = self.Bi_RNN([het_node.node_id], het_node.node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        post_neighbor = [item [1] for item in neighbor_order_n_p_u[:npu] if item[0] == 'p']
        user_neighbor = [item [1] for item in neighbor_order_n_p_u[:npu] if item[0] == 'u']
        news_neighbor = [item [1] for item in neighbor_order_n_p_u[:npu] if item[0] == 'n']
        if news_neighbor:
            n_agg_batch, n_aft_rnn_dict = self.SameType_Agg_Bi_RNN(news_neighbor, "news")
        if user_neighbor:
            u_agg_batch, u_aft_rnn_dict = self.SameType_Agg_Bi_RNN(user_neighbor, "user")
        if post_neighbor:
            p_agg_batch, p_aft_rnn_dict = self.SameType_Agg_Bi_RNN(post_neighbor, "post")
        encoder_emb_input = []
        decoder_emb_input = []
        encoder_order_input = []
        decoder_type_input = []
        decoder_order_input = []
        encoder_type_input = []
        encoder_order_input.append(0)
        encoder_emb_input.append(c_agg_batch[0])
        encoder_type_input.append(0)
        for i, type_id in enumerate(neighbor_order_n_p_u[:npu]):
            if type_id[0] == 'p':
                encoder_emb_input.append(torch.Tensor(p_aft_rnn_dict[type_id[1]]))
                encoder_type_input.append(1)
            elif type_id[0] == 'u':
                encoder_emb_input.append(torch.Tensor(u_aft_rnn_dict[type_id[1]]))
                encoder_type_input.append(2)
            else:
                encoder_emb_input.append(torch.Tensor(n_aft_rnn_dict[type_id[1]]))
                encoder_type_input.append(0)
            encoder_order_input.append(i+1)
        encoder_type_input = torch.LongTensor(encoder_type_input)
        encoder_order_input = torch.LongTensor(encoder_order_input)
        decoder_emb_input.append(c_agg_batch[0])
        decoder_type_input.append(0)
        for (i, type_id) in enumerate(news_neighbor):
            decoder_emb_input.append(torch.Tensor(n_aft_rnn_dict[type_id]))
            decoder_order_input.append(i+1)
            decoder_type_input.append(0)
        decoder_order_input = torch.LongTensor(decoder_order_input)
        decoder_type_input = torch.LongTensor(decoder_type_input)
        encoder_emb_input = torch.stack(encoder_emb_input)
        encoder_emb_input = encoder_emb_input.view(encoder_emb_input.shape[0], 1, encoder_emb_input.shape[1])
        decoder_emb_input = torch.stack(decoder_emb_input)
        decoder_emb_input = decoder_emb_input.view(decoder_emb_input.shape[0], 1, decoder_emb_input.shape[1])
        encoder_emb_input += self.type_encoder(encoder_type_input).view(encoder_emb_input.shape[0], 1, encoder_emb_input.shape[2])
        encoder_emb_input += self.pos_encoder(encoder_emb_input)
        decoder_emb_input += self.pos_encoder(decoder_emb_input)
        decoder_emb_input += self.type_encoder(decoder_type_input).view(decoder_emb_input.shape[0], 1, decoder_emb_input.shape[2])
        final_representation = self.transformer(encoder_emb_input, decoder_emb_input)
        return final_representation

    def output(self, c_embed_batch):
        batch_size = 1
        c_embed = c_embed_batch[0, 0, :]
        c_embed = c_embed.view(batch_size, 1, self.out_embed_d)
        c_embed = self.out_dropout(c_embed)
        c_embed_out = self.out_linear(c_embed)
        predictions = self.output_act(c_embed_out)  # log(1/(1+exp(-x)))    sigmoid = 1/(1+exp(-x))
        return predictions

    def forward(self, x, neighbor_order_n_p_u, neighbor_order_n):
        x = self.transformer_agg(x, neighbor_order_n_p_u, neighbor_order_n)
        x = self.output(c_embed_batch=x)
        return x


def BCELoss(predictions, true_label):
    loss = nn.BCELoss()
    predictions = predictions.view(1)
    tensor_label = torch.FloatTensor(np.array([true_label]))
    loss_sum = loss(predictions, tensor_label)
    return loss_sum


def save_checkpoint(model, optimizer, save_path, epoch, val_acc):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    return model, optimizer, epoch, val_acc


def train_test(data_real, data_fake, test_size):
    y_real = range(len(data_real))
    y_fake = range(len(data_fake))
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, y_real, test_size=test_size,
                                                                            random_state=42)
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, y_fake, test_size=test_size,
                                                                            random_state=42)
    np.savetxt('models/data_splits/PHEME/train_index_real.txt', y_train_real)
    np.savetxt('models/data_splits/PHEME/test_index_real.txt', y_test_real)
    np.savetxt('models/data_splits/PHEME/train_index_fake.txt', y_train_fake)
    np.savetxt('models/data_splits/PHEME/test_index_fake.txt', y_test_fake)
    return X_train_real, X_test_real, X_train_fake, X_test_fake


def load_train_test(data_real, data_fake, test_index_path_real='models/data_splits/PHEME/test_index_real.txt', test_index_path_fake='models/data_splits/PHEME/test_index_fake.txt'):
    a = np.loadtxt(test_index_path_real)
    a = a.astype('int32')
    b = np.loadtxt(test_index_path_fake)
    b = b.astype('int32')
    test_set_real = []
    train_set_real = []
    for i in range(len(data_real)):
        if i in a:
            test_set_real.append(data_real[i])
        else:
            train_set_real.append(data_real[i])
    test_set_fake = []
    train_set_fake = []
    for j in range(len(data_fake)):
        if j in b:
            test_set_fake.append(data_fake[j])
        else:
            train_set_fake.append(data_fake[j])
    return train_set_real, test_set_real, train_set_fake, test_set_fake


# split test set first
if path.exists('models/data_splits/PHEME/test_index_real.txt'):
    X_train_real, X_test_real, X_train_fake, X_test_fake = load_train_test(news_nodes_real, news_nodes_fake)
else:
    X_train_real, X_test_real, X_train_fake, X_test_fake = train_test(news_nodes_real, news_nodes_fake, 0.1)

# Shuffle the order in post nodes
train_val = X_train_real + X_train_fake
test_set = X_test_real + X_test_fake
np.random.shuffle(train_val)
np.random.shuffle(test_set)

# K-fold validation index
train_index = []
val_index = []
num_splits = 5
kfold = KFold(num_splits, True, 1)
for train, val in kfold.split(train_val):
    train_index.append(train)
    val_index.append(val)

# Initialize parameters
lrs = [0.001]
# lrs = [0.001]
batch_sizes = [2]
#outemb_dims = [200, 80, 112,160]
#hidden_dims = [64, 128, 256]
outemb_dims = [104]
hidden_dims = [64]
num_epoch = 50
num_folds = 5
base_PATH = 'models/pre-trained/PHEME/'
train_acc_set = []
train_loss_set = []
val_acc_set = []
val_loss_set = []


# In[5]:


import os
import random
print('Start training')
print('Total number of folds: ', num_folds)
#outemb_dims = [80]
# hidden_dims = [64]
num_folds = 1
print(lrs)
print(batch_sizes)
for index1, lr in enumerate(lrs):
    for index2, batch_size in enumerate(batch_sizes):
        for index3, out_dim in enumerate(outemb_dims):
            for index4, hidden_dim in enumerate(hidden_dims):
                print(lr)
                print(batch_size)
                print(out_dim)
                print(hidden_dim)
                PATH = base_PATH + 'lr'+ str(index1+1) + 'bs' + str(index2) + 'od' + str(index3) + 'hd' + str(index4) + '/'
                if not os.path.exists(PATH):
                    os.mkdir(PATH)
                for fold in range(num_folds):
                    print("Start for fold", fold + 1)
                    best_val = 0
                    running_loss = 0.0
                    val_loss = 0.0
                    test_loss = 0.0
                    best_epoch = 0
                    net = HetTransformer(input_dim=[768, 22], ini_hidden_dim=[300, 200], hidden_dim=100,
                          n_hidden_dim=64, n_ini_hidden_dim=128, n_output_dim=150,
                          u_hidden_dim=64, u_ini_hidden_dim=128, u_output_dim=150,
                          p_hidden_dim=64, p_ini_hidden_dim=128, p_output_dim=150,
                          out_embed_d=104, outemb_d=1, attn_heads=8, d_model=104, 
                          self_attn_heads=8, enc_layers=1, dec_layers=1)
                    net.init_weights()
                    print(net)
                    # Set up optimizer and scheduler
                    optimizer = optim.SGD(net.parameters(), lr=lr)
                    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)
                    for epoch in range(num_epoch):
                        print('Epoch:', epoch + 1)
                        m = 0.0
                        train_loss = 0.0
                        c = 0.0
                        running_loss = 0.0
                        v = 0.0
                        val_loss = 0.0
                        t = 0.0
                        test_loss = 0.0
                        real_count = 0.0
                        fake_count = 0.0
                        real_true = 0.0
                        fake_true = 0.0
                        # generate train and test set for current epoch
                        train_set = []
                        val_set = []
                        for t_index in train_index[fold]:
                            train_set.append(train_val[t_index])
                        for v_index in val_index[fold]:
                            val_set.append(train_val[v_index])
                        net.train()
                        random.shuffle(train_set)
                        for i in range(len(train_set)):
                            optimizer.zero_grad()
                            output = net(train_set[i], neighbor_dict[0][train_set[i].node_id], neighbor_dict[1][train_set[i].node_id])
                            out = output.item()
                            if (out/(1-out) >= ratio and train_set[i].label == 1) or (out/(1-out) < ratio and train_set[i].label == 0):
                                c += 1
                            cur_loss = BCELoss(predictions=output, true_label=train_set[i].label)
                            running_loss += cur_loss.item()
                            if i % 100 == 99:  # print every 100 mini-batches
                                print('Fold: %d, Epoch: %d, step: %5d, loss: %.4f, acc: %.4f' %
                                      (fold + 1, epoch + 1, i + 1, running_loss / 100, c / 100))
                                running_loss = 0.0
                                c = 0.0
                            if i % batch_size == 0:
                                loss = Variable(torch.zeros(1), requires_grad=True)
                            loss = loss + cur_loss
                            if i % batch_size == (batch_size - 1):
                                loss = loss / batch_size
                                loss.backward()
                                optimizer.step()
                        net.eval()
                        for j in range(len(val_set)):
                            output = net(val_set[j], neighbor_dict[0][val_set[j].node_id], neighbor_dict[1][val_set[j].node_id])
                            out = output.item()
                            if val_set[j].label == 1:
                                real_count += 1
                                if out/(1-out) >= ratio:
                                    real_true += 1
                            else:
                                fake_count += 1
                                if out/(1-out) < ratio:
                                    fake_true += 1
                            if (out/(1-out) >= ratio and val_set[j].label == 1) or (out/(1-out) < ratio and val_set[j].label == 0):
                                v += 1
                            vloss = BCELoss(predictions=output, true_label=val_set[j].label)
                            val_loss += vloss.item()

                        val_acc = v / len(val_set)
                        val_acc_set.append(val_acc)
                        val_loss_set.append(val_loss / len(val_set))
                        real_precision = real_true / (real_true + fake_count - fake_true)
                        fake_precision = fake_true / (fake_true + real_count - real_true)
                        real_recall = real_true / real_count
                        fake_recall = fake_true / fake_count
                        real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)
                        fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall)
                        print('Validation loss: %.4f, Validation accuracy: %.4f' % (val_loss / len(val_set), val_acc))
                        print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f' % (real_precision, real_recall, real_f1))
                        print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f' % (fake_precision, fake_recall, fake_f1))
                        if val_acc > best_val:
                            print('Update model at epoch:', epoch + 1)
                            cur_PATH = PATH + 'best_model' + '_' + str(fold) + '.tar'
                            save_checkpoint(net, optimizer, cur_PATH, epoch + 1, val_acc)
                            best_val = val_acc
                            best_epoch = epoch
                        scheduler.step()
                        if epoch - best_epoch >= 5:
                            break


# In[6]:


# Init net and optimizer skeletons
best_models = []
num_folds = 1
net = HetTransformer(input_dim=[768, 22], ini_hidden_dim=[300, 200], hidden_dim=100,
                          n_hidden_dim=64, n_ini_hidden_dim=128, n_output_dim=150,
                          u_hidden_dim=64, u_ini_hidden_dim=128, u_output_dim=150,
                          p_hidden_dim=64, p_ini_hidden_dim=128, p_output_dim=150,
                          out_embed_d=104, outemb_d=1, attn_heads=8, d_model=104, self_attn_heads=8, enc_layers=1, dec_layers=1)
net.init_weights()
optimizer = optim.SGD(net.parameters(), lr=lr)

PATH = 'models/pre-trained/PHEME/'
for count in range(num_folds):
    cur_PATH = PATH + 'best_model' + '_' + str(count) + '_82.5.tar'
    net, optimizer, epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)
    print(best_val)
    net.eval()
    best_models.append(net)

t = 0.0
test_loss = 0.0
real_count = 0.0
fake_count = 0.0
real_true = 0.0
fake_true = 0.0

for k in range(len(test_set)):
    output = 0.0
    avg_tloss = 0.0
    for fold in range(num_folds):
        result = best_models[fold](test_set[k], neighbor_dict[0][test_set[k].node_id], neighbor_dict[1][test_set[k].node_id])
        output += result.item()
        tloss = BCELoss(predictions=result, true_label=test_set[k].label)
        avg_tloss += tloss.item()
    output /= num_folds
    avg_tloss /= num_folds
    test_loss += avg_tloss
    if (output/(1-output) >= ratio and test_set[k].label == 1) or (output/(1-output) < ratio and test_set[k].label == 0):
        t += 1
    if test_set[k].label == 1:
        real_count += 1
        if output/(1-output) >= ratio:
            real_true += 1
    else:
        fake_count += 1
        if output/(1-output) < ratio:
            fake_true += 1

real_precision = real_true / (real_true + fake_count - fake_true)
fake_precision = fake_true / (fake_true + real_count - real_true)
real_recall = real_true / real_count
fake_recall = fake_true / fake_count
real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)
fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall)
print('Test loss: %.4f, Test accuracy: %.4f' % (test_loss / len(test_set), t / len(test_set)))
print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f' % (real_precision, real_recall, real_f1))
print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f' % (fake_precision, fake_recall, fake_f1))


# In[ ]:





# In[ ]:




