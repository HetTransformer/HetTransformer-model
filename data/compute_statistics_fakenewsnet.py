import os
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

in_dir = 'FakeNewsNet/graph_def'
edge_dirs = [
    os.path.join(in_dir, 'politifact', 'fake'),
    os.path.join(in_dir, 'politifact', 'real'),
    os.path.join(in_dir, 'gossipcop', 'fake'),
    os.path.join(in_dir, 'gossipcop', 'real'),
]
node_types = ['n', 'p', 'u']
edge_files = {
    ('n', 'n'): 'news-news edges.txt',
    ('n', 'p'): 'news-post edges.txt',
    ('p', 'u'): 'post-user edges.txt',
    ('u', 'u'): 'user-user edges.txt',
}
counts = {t1 : {t2 : [] for t2 in node_types} for t1 in node_types}
stats = {t1 : {t2 : dict() for t2 in node_types} for t1 in node_types}

adj_list = dict()
def add_adjacent(m, n):  # IN  adj_list['p123'] = ['u456', 'n789', ...]
    if m not in adj_list.keys():
        adj_list[m] = set()
    adj_list[m].add(n)


print("Read the graph...")
for edge_dir in edge_dirs:
    print("Reading", edge_dir)
    for (main_type, neig_type), edge_f in edge_files.items():
        with open(os.path.join(edge_dir, edge_f), "r") as f:
            for l in tqdm(f.readlines(), desc=main_type+' '+neig_type):
                l = l.strip().split()
                if len(l) != 2:
                    break  # gossipcop real does not have user edges for now
                add_adjacent(main_type + l[0], neig_type + l[1])
                add_adjacent(neig_type + l[1], main_type + l[0])


for node, neighbors in adj_list.items():
    for t in node_types:
        counts[node[0]][t].append(sum([1 for neig in neighbors if neig[0] == t]))


for t1 in node_types:
    for t2 in node_types:
        stats[t1][t2]['mean'] = np.mean(counts[t1][t2])
        stats[t1][t2]['std'] = np.std(counts[t1][t2])


plt.hist(counts['u']['p'])
plt.savefig('u-p.png')


print(json.dumps(stats, indent=4, sort_keys=True))
"""
Twitter graph definition statistics (i.e.跑random walk之前的)
{
    "n": {
        "n": 248.1491662324127,  # 平均1則news，連到248則news
        "p": 74.8747177349314,   # 平均1則news，連到75則posts
        "u": 0.0
    },
    "p": {
        "n": 1.0567749582920136, # 平均1則post，連到1.05則news (因為有些news沒有post！)
        "p": 0.0,
        "u": 1.315241894063553   # 平均1則post，連到1.31位users
    },
    "u": {
        "n": 0.0,
        "p": 2.7004044456301246, # 平均1位user，連到2.70則posts
        "u": 1.3022142580297684  # 平均1位user，連到1.30位users
    }
}
"""