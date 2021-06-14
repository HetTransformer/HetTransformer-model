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
        neighbor_dict_n_p_u[neighbor_list[0][1:-1]] = [(item[0], item[1:]) for item in neighbor_list[1:] if item[0] == 'p' or item[0] == 'u']
        neighbor_dict_n[neighbor_list[0][1:-1]] = [(item[0], item[1:]) for item in neighbor_list[1:] if item[0] == 'n']
        neighbor_dict_news[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'n']
        neighbor_dict_user[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'u']
        neighbor_dict_post[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'p']
    return neighbor_dict_n_p_u, neighbor_dict_n, neighbor_dict_news, neighbor_dict_post, neighbor_dict_user

neighbor_dict = neighbor_loader('/model/data_splits/GossipCop/n_neighbors.txt')
#print(list(neighbor_dict[2].values())[:5])
#print(list(neighbor_dict[3].values())[:5])
#print(list(neighbor_dict[4].values())[:5])

def data_loader(pathway='\', node_type="post"):
    if node_type == "news":
        
        news_node = []
        news_id = []
        news_label = []
        news_embed = []
        news_n_neigh = []
        news_p_neigh = []
        news_u_neigh = []
        for i in range(5):
            
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            if i == 4:
                for j in range(len(Lines)):
                    if j < len(Lines)-7:
                        if j % 7 == 0:
                            
                            _, id_, label = Lines[j].split()
                            news_id.append(id_)
                            news_label.append(int(label))
                            embed = []
                        if j % 7 == 1 or j % 7 == 2 or j % 7 == 3:
                            embed.append(list(map(float, Lines[j].split())))
                        if j % 7 == 3:
                            news_embed.append(embed)
            else:
                for j in range(len(Lines)):
                    if j < len(Lines):
                        if j % 7 == 0:
                            
                            _, id_, label = Lines[j].split()
                            news_id.append(id_)
                            news_label.append(int(label))
                            embed = []
                        if j % 7 == 1 or j % 7 == 2 or j % 7 == 3:
                            embed.append(list(map(float, Lines[j].split())))
                        if j % 7 == 3:
                            news_embed.append(embed)
                                        
            f.close()
        for i in range(len(news_id)):
            node = Het_Node(node_type="news", node_id=news_id[i], embed=news_embed[i],
                            label=news_label[i])
            news_node.append(node)
        # padding_node = Het_Node(node_type="news", node_id=padding_id, embed=padding_embed,
                            # neighbor_list_news=padding_news_n_neigh, neighbor_list_post=padding_news_p_neigh, neighbor_list_user=padding_news_u_neigh)
        return news_node# , padding_node

    elif node_type == 'post':
        post_node = []
        post_id = []
        post_embed = []
        for i in range(190):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j < len(Lines):
                    if j % 6 == 0:
                        id_ = Lines[j].split()
                        post_id.append(id_[1])
                        embed = []
                    if j % 6 == 1 or j % 6 == 2:
                        embed.append(list(map(float, Lines[j].split())))
                    if j % 6 == 2:
                        post_embed.append(embed)
                else:
                    if j % 6 == 0:
                        padding_id = 'pPADDING'
                        padding_embed = []
                    if j % 6 == 1 or j % 6 == 2:
                        padding_embed.append(list(map(float, Lines[j].split())))
            f.close()
        for i in range(len(post_id)):
            node = Het_Node(node_type="post", node_id=post_id[i], embed=post_embed[i])
            post_node.append(node)
        # padding_node = Het_Node(node_type='post', node_id=padding_id, embed=padding_embed)
        return post_node# , padding_node

    else:
        user_node = []
        user_id = []
        user_embed = []
        for i in range(85):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):               
                if j < len(Lines):
                    if j % 6 == 0:
                        id_ = Lines[j].split()
                        user_id.append(id_[1])
                        embed = []
                    if j % 6 == 1 or j % 6 == 2:
                        embed.append(list(map(float, Lines[j].split())))
                    if j % 6 == 2:
                        user_embed.append(embed)
                else:
                    if j % 6 == 0:
                        padding_id = 'uPADDING'
                        padding_embed = []
                    if j % 6 == 1 or j % 6 == 2:
                        padding_embed.append(list(map(float, Lines[j].split())))
            f.close()
        for i in range(len(user_id)):
            node = Het_Node(node_type="user", node_id=user_id[i], embed=user_embed[i])
            user_node.append(node)
        # padding_node = Het_Node(node_type='user', node_id=padding_id, embed=padding_embed)
        return user_node#, padding_node
news_nodes = data_loader(pathway='/FakeNewsNet/FNN_input/Gossipcop/gossipcop_200/normalized_news_nodes/', node_type="news")
post_nodes = data_loader(pathway='/FakeNewsNet/FNN_input/Gossipcop/gossipcop_200/normalized_post_nodes/', node_type="post")
user_nodes = data_loader(pathway='/FakeNewsNet/FNN_input/Gossipcop/gossipcop_200/normalized_user_nodes/', node_type="user")

news_emb_dict = {}
post_emb_dict = {}
user_emb_dict = {}

for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb
for news in news_nodes:
    news_emb_dict[news.node_id] = news.emb
    
#user_emb_dict[u_padding.node_id] = u_padding.emb
#post_emb_dict[p_padding.node_id] = p_padding.emb
#news_emb_dict[n_padding.node_id] = n_padding.emb

news_nodes_real = []
news_nodes_fake = []
for node in news_nodes:
    if node.label == 1:
        news_nodes_real.append(node)
    else:
        news_nodes_fake.append(node)
print("number of fake nodes: ", len(news_nodes_fake))
print("number of real nodes: ", len(news_nodes_real))
#fake : real = 0.9842 : 1

import math
from torch_position_embedding import PositionEmbedding
import random



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
        #print(x.shape)
        #print(self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NeighborTypeEmbedding(nn.Embedding):
    def __init__(self, num_node_type=3, embed_size=512):
        super().__init__(num_node_type, embed_size, padding_idx=0)

        
class Het_GNN(nn.Module):
    # features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim,
                 n_hidden_dim, n_ini_hidden_dim, n_output_dim,
                 u_hidden_dim, u_ini_hidden_dim, u_output_dim,
                 p_hidden_dim, p_ini_hidden_dim, p_output_dim,
                 out_embed_d, outemb_d=1, n_num_layers=1, u_num_layers=1, 
                 p_num_layers=1, num_layers=1, n_batch_size=1,
                 u_batch_size=1, p_batch_size=1, content_dict={}, 
                 n_rnn_type='LSTM', u_rnn_type='LSTM', p_rnn_type='LSTM',
                 rnn_type='LSTM', embed_d=200, symmetry=False, GCN_out2=300, 
                 GCN_out1=600, GCN_in=200, use_bias=True, attn_heads = 2, 
                 d_model = 512, feed_forward=1024, use_gcn = True, 
                 enc_layers=1, npu=15, self_attn_heads=2):
        super(Het_GNN, self).__init__()
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
        self.d_model = d_model
        self.feed_forward = feed_forward
        self.out_embed_d = out_embed_d
        # self.out_linear_d = out_linear
        self.outemb_d = outemb_d
        # self.features = features
        self.content_dict = content_dict
        self.GCN_in = out_embed_d
        self.GCN_out1 = GCN_out1
        self.GCN_out2 = out_embed_d * 2
        self.use_bias = use_bias
        self.symmetry = symmetry
        self.npu = npu
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=250)
        self.pos_decoder = PositionalEncoding(d_model, dropout=0.1, max_len=250)
        self.type_encoder = nn.Embedding(3, d_model, padding_idx=0)
        self.use_gcn = use_gcn
        self.n_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
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
        self.transformer = nn.Transformer(d_model=d_model, nhead=attn_heads, num_encoder_layers=enc_layers, num_decoder_layers=6, dim_feedforward=feed_forward, 
                                          dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=attn_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=enc_layers)
        # self.node_type_iput = NeighborTypeEmbedding(num_node_type=3, embed_size=self.d_model)
        # self.positional_embedding = PositionalEncoding(self.d_model, dropout=0.1, max_len=5000)
        # Define the initial linear hidden layer
        
        
        self.init_linear_text = nn.Linear(self.input_dim[0], self.hidden_dim)
        self.init_linear_image = nn.Linear(self.input_dim[1], self.hidden_dim)
        self.init_linear_other_p = nn.Linear(self.input_dim[2], self.hidden_dim)
        self.init_linear_other_user = nn.Linear(self.input_dim[3], self.hidden_dim)
        
        
#         self.init_linear_text = nn.Linear(self.input_dim[0], self.ini_hidden_dim[0])
#         self.init_linear_image = nn.Linear(self.input_dim[1], self.ini_hidden_dim[1])
#         self.init_linear_other = nn.Linear(self.input_dim[2], self.ini_hidden_dim[2])
#         self.init_linear_other_user = nn.Linear(self.input_dim[3], self.ini_hidden_dim[3])
        # Define the attention layer
        self.post_content_attention_other = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.news_title_LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
                                                           batch_first=True,
                                                           bidirectional=True, dropout=0.5)
        self.news_title_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.news_content_LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
                                                             batch_first=True,
                                                             bidirectional=True, dropout=0.5)
        self.news_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.post_content_LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
                                                             batch_first=True,
                                                             bidirectional=True, dropout=0.5)
        self.post_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.user_content_LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
                                                             batch_first=True,
                                                             bidirectional=True, dropout=0.5)
        self.user_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.LSTM_image = eval('nn.' + rnn_type)(self.ini_hidden_dim[1], self.hidden_dim, self.num_layers,
                                                 batch_first=True,
                                                 bidirectional=True, dropout=0.5)
        self.attention_image = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.LSTM_other = eval('nn.' + rnn_type)(self.ini_hidden_dim[2], self.hidden_dim, self.num_layers,
                                                 batch_first=True,
                                                 bidirectional=True, dropout=0.5)
        self.attention_other = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.LSTM_other_user = eval('nn.' + rnn_type)(self.ini_hidden_dim[3], self.hidden_dim, self.num_layers,
                                                 batch_first=True,
                                                 bidirectional=True, dropout=0.5)
        self.LSTM_other_user = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout = 0.2)
        self.layernorm1 = nn.LayerNorm([1,out_embed_d])
        self.layernorm2 = nn.LayerNorm([1,out_embed_d])
        self.layernorm3 = nn.LayerNorm([1,out_embed_d])
        self.layernorm4 = nn.LayerNorm([1,out_embed_d])
        self.layernorm5 = nn.LayerNorm([1,out_embed_d])
        # Define same_type_agg
        self.n_init_linear = nn.Linear(self.n_input_dim, self.n_hidden_dim)
        # self.n_init_linear = nn.Linear(self.n_input_dim, self.n_ini_hidden_dim)
        # self.u_init_dropout = nn.Dropout(p=0.2)
        self.n_LSTM = eval('nn.' + self.n_rnn_type)(self.n_ini_hidden_dim, self.n_hidden_dim, self.n_num_layers,
                                                    batch_first=True, bidirectional=True, dropout=0.5)
        self.n_attention = nn.MultiheadAttention(self.n_hidden_dim, self_attn_heads, dropout = 0.2)
        self.n_linear = nn.Linear(self.n_hidden_dim, self.n_output_dim)
        self.n_dropout = nn.Dropout(p=0.5)
        self.u_init_linear = nn.Linear(self.u_input_dim, self.u_hidden_dim)
        # self.u_init_linear = nn.Linear(self.u_input_dim, self.u_ini_hidden_dim)
        # self.u_init_dropout = nn.Dropout(p=0.2)
        self.u_LSTM = eval('nn.' + self.u_rnn_type)(self.u_ini_hidden_dim, self.u_hidden_dim, self.u_num_layers,
                                                    batch_first=True, bidirectional=True, dropout=0.5)
        self.u_attention = nn.MultiheadAttention(self.u_hidden_dim, self_attn_heads, dropout = 0.2)
        self.u_linear = nn.Linear(self.u_hidden_dim, self.u_output_dim)
        self.u_dropout = nn.Dropout(p=0.5)
        self.p_init_linear = nn.Linear(self.p_input_dim, self.p_hidden_dim)
        # self.p_init_linear = nn.Linear(self.p_input_dim, self.p_ini_hidden_dim)
        # self.p_init_dropout = nn.Dropout(p=0.2)
        self.p_LSTM = eval('nn.' + self.p_rnn_type)(self.p_ini_hidden_dim, self.p_hidden_dim, self.p_num_layers,
                                                    batch_first=True, bidirectional=True, dropout=0.5)
        self.p_attention = nn.MultiheadAttention(self.p_hidden_dim, self_attn_heads, dropout = 0.2)
        self.p_linear = nn.Linear(self.p_hidden_dim, self.p_output_dim)
        self.p_dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.out_dropout = nn.Dropout(p=0.25)
        self.out_linear = nn.Linear(self.out_embed_d, self.outemb_d)

        # self.out_final = nn.Linear(self.out_linear_d, self.outemb_d)
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
        nn.init.kaiming_uniform_(self.GCN_W1_user)
        nn.init.kaiming_uniform_(self.GCN_W1_post)
        nn.init.kaiming_uniform_(self.GCN_W1_news)
        nn.init.kaiming_uniform_(self.GCN_W2_user)
        nn.init.kaiming_uniform_(self.GCN_W2_post)
        nn.init.kaiming_uniform_(self.GCN_W2_news)
        if self.use_bias:
            nn.init.zeros_(self.user_bias1)
            nn.init.zeros_(self.post_bias1)
            nn.init.zeros_(self.news_bias1)
            nn.init.zeros_(self.user_bias2)
            nn.init.zeros_(self.post_bias2)
            nn.init.zeros_(self.news_bias2)

    def init_adj_degree(self, neighbor_id):
        adj_dim = len(neighbor_id)
        adj = np.ones((adj_dim, adj_dim))
        degree = np.diag(np.sum(adj, axis=0))
        return adj, degree

    def init_matrices(self, u_aft_rnn_dict, p_aft_rnn_dict, n_aft_rnn_dict):
        user_H = []
        post_H = []
        news_H = []
        user_H_d = u_aft_rnn_dict.values()
        post_H_d = p_aft_rnn_dict.values()
        news_H_d = n_aft_rnn_dict.values()
        for i in user_H_d:
            i = i.view(1, i.shape[0])
            user_H.append(i)
        for j in post_H_d:
            j = j.view(1, j.shape[0])
            post_H.append(j)
        for k in news_H_d:
            k = k.view(1, k.shape[0])
            news_H.append(k)
        user_A, user_D = self.init_adj_degree(u_aft_rnn_dict)
        post_A, post_D = self.init_adj_degree(p_aft_rnn_dict)
        news_A, news_D = self.init_adj_degree(n_aft_rnn_dict)
        user_H = torch.cat(user_H, dim=0)
        post_H = torch.cat(post_H, dim=0)
        news_H = torch.cat(news_H, dim=0)
        user_A = torch.Tensor(user_A)
        user_D = torch.Tensor(user_D)
        post_A = torch.Tensor(post_A)
        post_D = torch.Tensor(post_D)
        news_A = torch.Tensor(news_A)
        news_D = torch.Tensor(news_D)
        return user_A, user_D, user_H, post_A, post_D, post_H, news_A, news_D, news_H

    def GCN_layer(self, user_A, user_D, user_H, post_A, post_D, post_H, news_A, news_D, news_H, layer_index):
        if self.symmetry == False:
            user_in = torch.mm(torch.inverse(user_D), user_A)
            post_in = torch.mm(torch.inverse(post_D), post_A)
            news_in = torch.mm(torch.inverse(news_D), news_A)
        else:
            user_D_head = fractional_matrix_power(user_D, -0.5)
            post_D_head = fractional_matrix_power(post_D, -0.5)
            news_D_head = fractional_matrix_power(news_D, -0.5)
            user_D_head = torch.Tensor(user_D_head)
            post_D_head = torch.Tensor(post_D_head)
            news_D_head = torch.Tensor(news_D_head)
            user_in = torch.mm(torch.mm(user_D_head, user_A), user_D_head)
            post_in = torch.mm(torch.mm(post_D_head, post_A), post_D_head)
            news_in = torch.mm(torch.mm(news_D_head, news_A), news_D_head)
        user_in_features = torch.mm(user_in, user_H)
        post_in_features = torch.mm(post_in, post_H)
        news_in_features = torch.mm(news_in, news_H)
        if layer_index == 1:
            user_output = torch.mm(user_in_features, self.GCN_W1_user)
            post_output = torch.mm(post_in_features, self.GCN_W1_post)
            news_output = torch.mm(news_in_features, self.GCN_W1_news)
            if self.use_bias:
                user_output += self.user_bias1
                post_output += self.post_bias1
                news_output += self.news_bias1
        else:
            user_output = torch.mm(user_in_features, self.GCN_W2_user)
            post_output = torch.mm(post_in_features, self.GCN_W2_post)
            news_output = torch.mm(news_in_features, self.GCN_W2_news)
            if self.use_bias:
                user_output += self.user_bias2
                post_output += self.post_bias2
                news_output += self.news_bias2
        return user_output, post_output, news_output

    def GCN_net(self, u_aft_rnn_dict, p_aft_rnn_dict, n_aft_rnn_dict):
        user_A, user_D, user_H, post_A, post_D, post_H, news_A, news_D, news_H = self.init_matrices(u_aft_rnn_dict,
                                                                                                    p_aft_rnn_dict,
                                                                                                    n_aft_rnn_dict)
        gcn1_user, gcn1_post, gcn1_news = self.GCN_layer(user_A, user_D, user_H, post_A, post_D, post_H, news_A, news_D,
                                                         news_H, 1)
        gcn1_user = self.relu(gcn1_user)
        gcn1_post = self.relu(gcn1_post)
        gcn1_news = self.relu(gcn1_news)
        gcn2_user, gcn2_post, gcn2_news = self.GCN_layer(user_A, user_D, gcn1_user, post_A, post_D, gcn1_post, news_A,
                                                         news_D, gcn1_news, 2)
        gcn2_user = torch.mean(gcn2_user, 0)
        gcn2_post = torch.mean(gcn2_post, 0)
        gcn2_news = torch.mean(gcn2_news, 0)
        gcn2_user = gcn2_user.view(1, gcn2_user.shape[0])
        gcn2_post = gcn2_post.view(1, gcn2_post.shape[0])
        gcn2_news = gcn2_news.view(1, gcn2_news.shape[0])
        gcn2_user = self.softmax(gcn2_user)
        gcn2_post = self.softmax(gcn2_post)
        gcn2_news = self.softmax(gcn2_news)
        return gcn2_user, gcn2_post, gcn2_news

    def Bi_RNN(self, neighbor_id, node_type, post_emb_dict, user_emb_dict, news_emb_dict):
        # Forward pass through initial hidden layer
        new_id = []
        if node_type == "news":
            input_title = []
            input_content = []
            input_image = []
            for i in neighbor_id:
                if ("news", i) not in self.content_dict:
                    input_title.append(news_emb_dict[i][0])
                    input_content.append(news_emb_dict[i][1])
                    input_image.append(news_emb_dict[i][2])
                    new_id.append(i)
            input_title = torch.Tensor(input_title)
            input_image = torch.Tensor(input_image)
            input_content = torch.Tensor(input_content)
            linear_input_title = self.init_linear_text(input_title)
            linear_input_content = self.init_linear_text(input_content)
            linear_input_image = self.init_linear_image(input_image)
            linear_input_title = linear_input_title.view(linear_input_title.shape[0], 1, linear_input_title.shape[1])
            linear_input_content = linear_input_content.view(linear_input_content.shape[0], 1,
                                                             linear_input_content.shape[1])
            linear_input_image = linear_input_image.view(linear_input_image.shape[0], 1, linear_input_image.shape[1])
            attention_out_title, self.hidden_text = self.news_title_attention_text(linear_input_title, linear_input_title, linear_input_title)
            attention_out_content, self.hidden_text = self.news_content_attention_text(linear_input_content, linear_input_content, linear_input_content)
            attention_out_image, self.hidden_image = self.attention_image(linear_input_image, linear_input_image, linear_input_image)
            attention_out_title = self.layernorm1(attention_out_title)
            attention_out_content = self.layernorm2(attention_out_content)
            concate = torch.cat((attention_out_title, attention_out_content, attention_out_image), 1)
        if node_type == "post":
            input_a = []
            input_b = []
            for i in neighbor_id:
                if ("post", i) not in self.content_dict:
                    input_a.append(post_emb_dict[i][1])
                    input_b.append(post_emb_dict[i][0])
                    new_id.append(i)
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_a)
            linear_input_other = self.init_linear_other_p(input_b)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            attention_out_text, self.hidden_text = self.post_content_attention_text(linear_input_text, linear_input_text, linear_input_text)
            attention_out_other, self.hidden_other = self.post_content_attention_other(linear_input_other, linear_input_other, linear_input_other)
            attention_out_text = self.layernorm3(attention_out_text)
            attention_out_other = self.layernorm3(attention_out_other)
            # LSTM_out_image, self.hidden_image = self.LSTM_image(linear_input_image)
            concate = torch.cat((attention_out_text, attention_out_other), 1)
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
            linear_input_text = self.init_linear_text(input_b)
            linear_input_other = self.init_linear_other_user(input_a)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            attention_out_text, self.hidden_text = self.user_content_attention_text(linear_input_text, linear_input_text, linear_input_text)
            attention_out_other, self.hidden_other = self.attention_other(linear_input_other, linear_input_other, linear_input_other)
            attention_out_text = self.layernorm4(attention_out_text)
            attention_out_other = self.layernorm5(attention_out_other)
            concate = torch.cat((attention_out_text, attention_out_other), 1)

        # mean pooling all the states
        mean_pooling = torch.mean(concate, 1)

        return mean_pooling
    
    def het_Bi_RNN(self, neighbor_id, node_type, post_emb_dict, user_emb_dict, news_emb_dict):
        # Forward pass through initial hidden layer
        new_id = []
        if node_type == "news":
            input_title = []
            input_content = []
            input_image = []
            for i in neighbor_id:
                if ("news", i) not in self.content_dict:
                    input_title.append(news_emb_dict[i][0])
                    input_content.append(news_emb_dict[i][1])
                    input_image.append(news_emb_dict[i][2])
                    new_id.append(i)
            input_title = torch.Tensor(input_title)
            input_image = torch.Tensor(input_image)
            input_content = torch.Tensor(input_content)
            linear_input_title = self.init_linear_text(input_title)
            linear_input_content = self.init_linear_text(input_content)
            linear_input_image = self.init_linear_image(input_image)
            linear_input_title = linear_input_title.view(linear_input_title.shape[0], 1, linear_input_title.shape[1])
            linear_input_content = linear_input_content.view(linear_input_content.shape[0], 1,
                                                             linear_input_content.shape[1])
            linear_input_image = linear_input_image.view(linear_input_image.shape[0], 1, linear_input_image.shape[1])
            attention_out_title, self.hidden_text = self.news_title_LSTM_text(linear_input_title)
            attention_out_content, self.hidden_text = self.news_content_LSTM_text(linear_input_content)
            attention_out_image, self.hidden_image = self.LSTM_image(linear_input_image)
            attention_out_title = self.layernorm1(attention_out_title)
            attention_out_content = self.layernorm2(attention_out_content)
            concate = torch.cat((attention_out_title, attention_out_content, attention_out_image), 1)
        if node_type == "post":
            input_a = []
            for i in neighbor_id:
                if ("post", i) not in self.content_dict:
                    input_a.append(post_emb_dict[i][1])
                    # input_b.append(post_emb_dict[i][1])
                    new_id.append(i)
            input_a = torch.Tensor(input_a)
            # input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_a)
            # linear_input_image = self.init_linear_image(input_b)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            # linear_input_image = linear_input_image.view(linear_input_image.shape[0], 1, linear_input_image.shape[1])
            attention_out_text, self.hidden_text = self.post_content_attention_text(linear_input_text)
            attention_out_text = self.layernorm3(attention_out_text)
            # LSTM_out_image, self.hidden_image = self.LSTM_image(linear_input_image)
            concate = attention_out_text
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
            linear_input_text = self.init_linear_text(input_b)
            linear_input_other = self.init_linear_other_user(input_a)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            attention_out_text, self.hidden_text = self.user_content_attention_text(linear_input_text, linear_input_text, linear_input_text)
            attention_out_other, self.hidden_other = self.attention_other(linear_input_other)
            attention_out_text = self.layernorm4(attention_out_text)
            attention_out_other = self.layernorm5(attention_out_other)
            concate = torch.cat((attention_out_text, attention_out_other), 1)

        # mean pooling all the states
        mean_pooling = torch.mean(concate, 1)

        return mean_pooling

    # features: list of [(id)]
    def transformer_aggregator(self, node, neighbor_ids, neighbor_node_types):
        neighbor_content_embedings = self.het_Bi_RNN(neighbor_id, node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        decoder_input = neighbor_content_embedings.view(neighbor_content_embedings.shape[0], 1, neighbor_content_embedings.shape[1])
        self_content_embedings = self.het_Bi_RNN([node.node_id], [node.node_type], post_emb_dict, user_emb_dict, news_emb_dict)
        self_content_embedings = self_content_embedings.view(self_content_embedings.shape[0], 1, self_content_embedings.shape[1])
        encoder_input = self.node_type_iput(neighbor_node_types) + self.positional_embedding(neighbor_content_embedings)
        output = self.transformer(encoder_input, decoder_input) 
        return output
        
    def SameType_Agg_Bi_RNN(self, neighbor_id, node_type):
        content_embedings = self.Bi_RNN(neighbor_id, node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        aft_rnn_dict = {}
        for i in range(len(neighbor_id)):
            aft_rnn_dict[neighbor_id[i]] = content_embedings[i]
        if node_type == 'news':
            linear_input = self.n_init_linear(content_embedings)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            attention_out, hidden = self.n_attention(linear_input, linear_input, linear_input)
            #LSTM_out = self.m6(LSTM_out)
            last_state = self.n_linear(attention_out)
            last_state = self.n_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling, aft_rnn_dict
        if node_type == 'post':
            linear_input = self.p_init_linear(content_embedings)
            # linear_input = self.p_init_dropout(linear_input)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            attention_out, hidden = self.p_attention(linear_input, linear_input, linear_input)
            #LSTM_out = self.m7(LSTM_out)
            last_state = self.p_linear(attention_out)
            last_state = self.p_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling, aft_rnn_dict
        if node_type == 'user':
            linear_input = self.u_init_linear(content_embedings)
            # linear_input = self.u_init_dropout(linear_input)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            attention_out, hidden = self.u_attention(linear_input, linear_input, linear_input)
            last_state = self.u_linear(attention_out)
            last_state = self.u_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling, aft_rnn_dict

    
    def transformer_agg(self, het_node, neighbor_order_n_p_u, neighbor_order_n, npu=200, n=5):  # heterogeneous neighbor aggregation
        # attention module
        c_agg_batch = self.Bi_RNN([het_node.node_id], het_node.node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        # news_neighbor_decoder = [item [1] for item in neighbor_order_n[:n]]
        post_neighbor = [item [1] for item in neighbor_order_n_p_u[:npu] if item[0] == 'p']
        user_neighbor = [item [1] for item in neighbor_order_n_p_u[:npu] if item[0] == 'u']
        news_neighbor = [item [1] for item in neighbor_order_n_p_u[:npu] if item[0] == 'n']
        # news_neighbor = [item [1] for item in neighbor_order_n_p_u if item[0] == 'n']
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
        for i, type_id in enumerate(neighbor_order_n_p_u):
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
        #encoder_emb_input = torch.Tensor(encoder_emb_input)
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
    
    def node_het_agg(self, het_node):  # heterogeneous neighbor aggregation

        # attention module
        c_agg_batch = self.Bi_RNN([het_node.node_id], het_node.node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        n_agg_batch, n_aft_rnn_dict = self.SameType_Agg_Bi_RNN(het_node.neighbors_news, "news")
        u_agg_batch, u_aft_rnn_dict = self.SameType_Agg_Bi_RNN(het_node.neighbors_user, "user")
        p_agg_batch, p_aft_rnn_dict = self.SameType_Agg_Bi_RNN(het_node.neighbors_post, "post")
        if self.use_gcn:
            gcn2_user, gcn2_post, gcn2_news = self.GCN_net(u_aft_rnn_dict, p_aft_rnn_dict, n_aft_rnn_dict)

        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        n_agg_batch_2 = torch.cat((c_agg_batch, n_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        u_agg_batch_2 = torch.cat((c_agg_batch, u_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        if self.use_gcn:
            u_agg_batch_2 = torch.cat((gcn2_user, u_agg_batch_2), 0)
            u_agg_batch_2 = torch.mean(u_agg_batch_2, 0, keepdims=True)

            p_agg_batch_2 = torch.cat((gcn2_post, p_agg_batch_2), 0)
            p_agg_batch_2 = torch.mean(p_agg_batch_2, 0, keepdims=True)

            n_agg_batch_2 = torch.cat((gcn2_news, n_agg_batch_2), 0)
            n_agg_batch_2 = torch.mean(n_agg_batch_2, 0, keepdims=True)

        # compute weights
        concate_embed = torch.cat((c_agg_batch_2, u_agg_batch_2, p_agg_batch_2, n_agg_batch_2), 1).view(
            len(c_agg_batch), 4,
            self.embed_d * 2)
        if het_node.node_type == "user":
            atten_w = self.act(torch.bmm(concate_embed, self.u_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.u_neigh_att.size())))
        elif het_node.node_type == "post":
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.p_neigh_att.size())))
        else:
            atten_w = self.act(torch.bmm(concate_embed, self.n_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.n_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)
        # print(atten_w)

        # weighted combination
        concate_embed = torch.cat((c_agg_batch, u_agg_batch, p_agg_batch, n_agg_batch), 1).view(len(c_agg_batch), 4,
                                                                                                self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

        return weight_agg_batch

    def output(self, c_embed_batch):
        batch_size = 1
        # make c_embed 3D tensor. Batch_size * 1 * embed_d
        #print(c_embed_batch.shape)
        c_embed = c_embed_batch[0, 0, :]
        c_embed = c_embed.view(batch_size, 1, self.out_embed_d)
        c_embed = self.out_dropout(c_embed)
        c_embed_out = self.out_linear(c_embed)
        # c_embed_out = self.out_final(c_embed)
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
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, y_real, test_size=test_size, random_state=42)
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, y_fake, test_size=test_size, random_state=42)
    np.savetxt('/model/data_splits/GossipCop/train_index_real.txt', y_train_real)
    np.savetxt('/model/data_splits/GossipCop/test_index_real.txt', y_test_real)
    np.savetxt('/model/data_splits/GossipCop/train_index_fake.txt', y_train_fake)
    np.savetxt('/model/data_splits/GossipCop/test_index_fake.txt', y_test_fake)
    return X_train_real, X_test_real, X_train_fake, X_test_fake



def load_train_test(data_real, data_fake, test_index_path_real='/model/data_splits/GossipCop/test_index_real.txt', test_index_path_fake = '/model/data_splits/GossipCop/test_index_fake.txt'):
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
if path.exists('/model/data_splits/GossipCop/test_index_real.txt'):
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

npus = [2,4,8,16,32,64]
# Initialize parameters
lr = 0.001
num_epoch = 40
num_folds = 1
batch_size = 4
PATH = '/model/best_models/GossipCop/'


print('Start training')

best_val = 0
for npu in npus:
    print('number of neighbors: ', npu)
    for fold in range(num_folds):
        print("Start for fold", fold + 1)
        
        running_loss = 0.0
        val_loss = 0.0
        test_loss = 0.0
        best_epoch = 0
        net = Het_GNN(input_dim=[768, 512, 3, 29], ini_hidden_dim=[200, 200, 200, 200], hidden_dim=100,
                      n_hidden_dim=200, n_ini_hidden_dim=128, n_output_dim=150,
                      u_hidden_dim=200, u_ini_hidden_dim=128, u_output_dim=150,
                      p_hidden_dim=200, p_ini_hidden_dim=128, p_output_dim=150,
                      out_embed_d=208, GCN_out1=550, d_model=208, attn_heads=8, 
                      use_gcn=True, enc_layers=1, feed_forward=512, npu=npu)
        net.init_weights()
        print(net)
        # Set up optimizer and scheduler
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=4, gamma=0.01)
        # net, optimizer, epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)
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
            random.shuffle(train_set)
            net.train()
            for i in range(len(train_set)):
                optimizer.zero_grad()
                output = net(train_set[i], neighbor_dict[0][train_set[i].node_id], neighbor_dict[1][train_set[i].node_id])
                if (output.item() >= 0.5 and train_set[i].label == 1) or (output.item() < 0.5 and train_set[i].label == 0):
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
                if val_set[j].label == 1:
                    real_count += 1
                    if output.item() >= 0.5:
                        real_true += 1
                else:
                    fake_count += 1
                    if output.item() < 0.5:
                        fake_true += 1
                if (output.item() >= 0.5 and val_set[j].label == 1) or (output.item() < 0.5 and val_set[j].label == 0):
                    v += 1
                vloss = BCELoss(predictions=output, true_label=val_set[j].label)
                val_loss += vloss.item()
            val_acc = v / len(val_set)
            #val_acc_set.append(val_acc)
            #val_loss_set.append(val_loss / len(val_set))
            real_precision = real_true / (real_true + fake_count - fake_true)
            fake_precision = fake_true / (fake_true + real_count - real_true)
            real_recall = real_true / real_count
            fake_recall = fake_true / fake_count
            real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)
            fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall)
            print('Validation loss: %.4f, Validation accuracy: %.4f' % (val_loss / len(val_set), val_acc))
            print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f' % (real_precision, real_recall, real_f1))
            print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f' % (fake_precision, fake_recall, fake_f1))
                        if val_acc >= best_val:
                print('Update model at epoch:', epoch + 1)
                cur_PATH = PATH + 'best_model' + '_' + str(npu) + '.tar'
                save_checkpoint(net, optimizer, cur_PATH, epoch + 1, val_acc)
                best_val = val_acc
                best_epoch = epoch
            scheduler.step()
            if epoch - best_epoch >= 5:
                break
        print('Finish training')

    print('==============================================================')

t = 0.0
test_loss = 0.0
real_count = 0.0
fake_count = 0.0
real_true = 0.0
fake_true = 0.0
net = Het_GNN(input_dim=[768, 512, 3, 29], ini_hidden_dim=[200, 200, 200, 200], hidden_dim=100,
                      n_hidden_dim=200, n_ini_hidden_dim=128, n_output_dim=150,
                      u_hidden_dim=200, u_ini_hidden_dim=128, u_output_dim=150,
                      p_hidden_dim=200, p_ini_hidden_dim=128, p_output_dim=150,
                      out_embed_d=208, GCN_out1=550, d_model=208, attn_heads=8, 
                      use_gcn=True, enc_layers=1, feed_forward=512, npu=npu)
optimizer = optim.SGD(net.parameters(), lr=lr)
for npu in npus:
    cur_PATH = PATH + 'best_model' + '_' + str(npu) + '.tar'
    net, optimizer, epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)
    net.eval()
    for k in range(len(test_set)):
        output = 0.0
        avg_tloss = 0.0
        for fold in range(num_folds):
            result = net(test_set[k], neighbor_dict[0][test_set[k].node_id], neighbor_dict[1][test_set[k].node_id])
            output += result.item()
            tloss = BCELoss(predictions=result, true_label=test_set[k].label)
            avg_tloss += tloss.item()
        output /= num_folds
        avg_tloss /= num_folds
        test_loss += avg_tloss
        if (output >= 0.5 and test_set[k].label == 1) or (output < 0.5 and test_set[k].label == 0):
            t += 1
        if test_set[k].label == 1:
            real_count += 1
            if output >= 0.5:
                real_true += 1
        else:
            fake_count += 1
            if output < 0.5:
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


