"""
* All edges are undirected.
* All IDs are string.
"""
import random
from collections import Counter
from tqdm import tqdm
import os
from multiprocessing import Manager, Pool

def recompute_involved(nei_list):
    involved = {t : set() for t in node_types}
    if typed_rwr:
        for node_1, nei_d in tqdm(nei_list.items(), desc='recompute involved'):
            involved[node_1[0]].add(node_1)
            for nodes in nei_d.values():
                for node_2 in nodes:
                    involved[node_2[0]].add(node_2)
    else:
        for node_1, nodes in tqdm(nei_list.items(), desc='recompute involved'):
            involved[node_1[0]].add(node_1)
            for node_2 in nodes:
                involved[node_2[0]].add(node_2)
    return involved

def rwr_worker(start_node, nei_list_subsets, desc, j, nodes_len, typed_rwr, no_rwr, max_steps, max_neigh, restart_rate, adj_list):
    # typed_rwr=True  -> OUT nei_list['p123']['u'] = ['u456', 'u789', ...]
    # typed_rwr=False -> OUT nei_list['p123'] = ['u456', 'n789', ...]
    nei_list = {start_node : {t : [] for t in node_types}} if typed_rwr \
        else {start_node : []}

    def try_add_neighbor(start_node, cur_node, num_neighs):
        t = cur_node[0]
        if not typed_rwr:
            nei_list[start_node].append(cur_node)
            return num_neighs + 1
        elif len(nei_list[start_node][t]) < min_neigh[t] or \
                all([len(nei_list[start_node][s]) >= min_neigh[s] for s in node_types if s != t]):
            nei_list[start_node][t].append(cur_node)
            return num_neighs + 1
        else:
            return num_neighs

    def get_top_k_most_frequent(neighbors, k, exclude):
        counter = Counter(neighbors)
        counter.pop(exclude, None)
        items = sorted(list(counter.items()), key=lambda x: -x[1])
        neighbors[:] = [items[i][0] for i in range(min(k, len(items)))]

    def enforce_edges(top_k, node, nn, k):
        for neig in adj_list[node]:
            if neig[0] == nn and neig not in top_k:
                top_k.insert(0, neig)
        top_k[:] = top_k[:k]

    def write_neighbor(node):
        if typed_rwr:
            for nn in node_types:
                get_top_k_most_frequent(nei_list[node][nn], max_uniq_neigh[nn], exclude=node)
                if (node[0], nn) in edges_to_enforce or (nn, node[0]) in edges_to_enforce:
                    enforce_edges(nei_list[node][nn], node, nn, max_uniq_neigh[nn])
        else:
            get_top_k_most_frequent(nei_list[node], max_neigh, exclude=node)
    
    num_neighs = 0
    if no_rwr:
        if len(adj_list[start_node]) <= max_neigh:
            nei_list[start_node] = adj_list[start_node]
        else:
            nei_list[start_node] = random.sample(adj_list[start_node], max_neigh)
    else:
        cur_node = start_node
        steps = 0
        while steps < max_steps:
            rand_p = random.random()
            if rand_p < restart_rate:
                cur_node = start_node
            else:
                cur_node = random.choice(adj_list[cur_node])
                num_neighs = try_add_neighbor(start_node, cur_node, num_neighs)
            steps += 1
        write_neighbor(start_node)
    
    nei_list_subsets.append(nei_list)
    if j % 1000 == 0:
        print(desc, '{:7} {:7} {:.4}'.format(j, nodes_len, j/nodes_len))

def save_result_worker(nei_list, involved, t, return_dict):
    written = 0
    with open(os.path.join(output_dir, f'{t}_neighbors.txt'), 'w') as f:
        if typed_rwr:
            for node, type_neighs in tqdm(nei_list.items(), desc=f'write {t} neigh'):
                if node[0] == t:
                    if all([len(type_neighs[t]) == 0 for t in node_types]):
                        continue
                    f.write(node + ':')
                    for neig_type in node_types:
                        f.write(' ' + ' '.join(type_neighs[neig_type]))
                        f.write((' ' + neig_type + 'PADDING') * (
                            max(0, max_uniq_neigh[neig_type] - len(type_neighs[neig_type]))
                        ))
                        written += len(type_neighs[neig_type])
                    f.write('\n')
        else:
            for node, node_neighs in tqdm(nei_list.items(), desc=f'write {t} neigh'):
                if node[0] == t:
                    if len(node_neighs) == 0:
                        continue
                    f.write(f"{node}: {' '.join(node_neighs)}\n")
                    written += len(node_neighs)
    with open(os.path.join(output_dir, f'{t}_involved.txt'), "w") as f:
        f.write(' '.join(list(involved[t])) + "\n")
    ret_str = "type {}: {:10} neighbors written.\n".format(t, written) + \
              "        {:10} nodes involved.\n".format(len(involved[t]))
    return_dict[t] = ret_str

def random_walk_with_restart():
    nei_list, nodes = dict(), dict()
    nodes = {t : set() for t in node_types}     # IN  nodes['p'] = {'p123', 'p456', ...}
    involved = {t : set() for t in node_types}  # OUT involved['p'] = {'p123', 'p456', ...}
    manager = Manager()

    def add_adjacent(m, n):
        if m not in adj_list.keys():
            adj_list[m] = []
        adj_list[m].append(n)

    def update_nei_list_subsets(nei_list, nei_list_subsets):
        for nei_list_subset in tqdm(nei_list_subsets, 'update_nei_list_subsets'):
            nei_list.update(nei_list_subset)
            del nei_list_subset

    def rwr(nodes_set, desc):
        nodes_list = list(nodes_set)
        nei_list_subsets = manager.list()
        with Pool(num_process) as p:
            p.starmap(rwr_worker, [(nodes_list[i], nei_list_subsets, desc, i, len(nodes_list), typed_rwr, no_rwr, max_steps, max_neigh, restart_rate, adj_list) for i in range(len(nodes_list))])
        update_nei_list_subsets(nei_list, nei_list_subsets)
    
    def compute_stats():
        stats = {t1: {t2: [] for t2 in node_types} for t1 in node_types}
        for n1, v in tqdm(nei_list.items(), desc='compute_stats'):
            if typed_rwr:
                for t2, x in v.items():
                    stats[n1[0]][t2].append(len(x))
            else:
                for t in node_types:
                    stats[n1[0]][t].append(0)
                for n2 in v:
                    stats[n1[0]][n2[0]][-1] += 1
        stats_str = []
        for t1 in node_types:
            for t2 in node_types:
                stats_str.append('stats {} {} {:.6f}'.format(
                    t1, t2,
                    sum(stats[t1][t2]) / len(stats[t1][t2]) if len(stats[t1][t2]) > 0 else 0))
        return stats_str
    
    def save_result(t):
        return_dict = dict()
        save_result_worker(nei_list, involved, t, return_dict)
        return [_ for _ in return_dict.values()]

    print("Reading", edge_dir)
    for (main_type, neig_type), edge_f in edge_files.items():
        with open(os.path.join(edge_dir, edge_f), "r") as f:
            for l in tqdm(f.readlines(), desc='read ' + main_type+' '+neig_type):  ########################################
                l = l.strip().split()
                add_adjacent(main_type + l[0], neig_type + l[1])
                add_adjacent(neig_type + l[1], main_type + l[0])
                nodes[main_type].add(main_type + l[0])
                nodes[neig_type].add(neig_type + l[1])

    print("Each node takes turns to be the starting node...")
    strs = []
    for node_type in node_types:
        rwr(nodes[node_type], node_type + ' rwr')
        involved = recompute_involved(nei_list)
        strs.extend(save_result(node_type))
    
    strs.extend(compute_stats())
    for s in strs:
        print(s)
    with open(os.path.join(output_dir, f'stats.txt'), 'w') as f:
        f.write('\n'.join(strs) + '\n')


if __name__ == "__main__":
    max_steps = 10000
    max_neigh = 512
    num_process = 4
    restart_rate = 0.5
    # dataset = 'politifact'  # 'politifact', 'pheme', 'buzzfeed'

    no_rwr = False
    typed_rwr = False
    sensitivity_test = False
    assert sum([typed_rwr, sensitivity_test, no_rwr]) <= 1  # Choose at most one mode

    for dataset in ['politifact', 'gossipcop', 'pheme',]:
        if dataset in ['politifact', 'gossipcop']:
            prefix = f'fnn_{dataset}_'
            in_dir = 'FakeNewsNet/graph_def'
            edge_dir = os.path.join(in_dir, dataset)
            node_types = ['n', 'p', 'u']
            edge_files = {
                ('n', 'n'): 'news-news edges.txt',
                ('n', 'p'): 'news-post edges.txt',
                ('p', 'u'): 'post-user edges.txt',
                ('u', 'u'): 'user-user edges.txt',
            }
            edges_to_enforce = {('p', 'u'),}
        elif dataset == 'pheme':
            prefix = 'pheme_'
            in_dir = 'pheme-figshare'
            edge_dir = in_dir
            node_types = ['n', 'p', 'u']
            edge_files = {
                # ('n', 'n'): 'PhemeNewsNews.txt',
                ('n', 'p'): 'PhemeNewsPost.txt',
                ('n', 'u'): 'PhemeNewsUser.txt',
                ('p', 'p'): 'PhemePostPost.txt',
                ('p', 'u'): 'PhemePostUser.txt',
                ('u', 'u'): 'PhemeUserUser.txt',
            }
            edges_to_enforce = {('n', 'u'), ('p', 'u'),}
        elif dataset == 'buzzfeed':
            prefix = 'buzzfeed_'
            in_dir = 'buzzfeed-kaggle'
            edge_dir = in_dir
            node_types = ['n', 's', 'u']
            edge_files = {
                ('s', 'n'): 'BuzzFeedSourceNews.txt',
                ('n', 'n'): 'BuzzFeedNewsNews.txt',
                ('n', 'u'): 'BuzzFeedNewsUser.txt',
                ('u', 'u'): 'BuzzFeedUserUser.txt',
            }
            edges_to_enforce = {('n', 'u'),}
        else:  # "weibo"
            raise NotImplementedError

        if typed_rwr:
            min_neigh = {node_type: 1000 if node_type == 'u' else 50 for node_type in node_types}
            max_uniq_neigh = {node_type: 100 if node_types == 'u' else 5 for node_type in node_types}
            configuration_tag = prefix + '_'.join([f'{k}{max_uniq_neigh[k]}' for k in node_types])
        elif no_rwr:
            configuration_tag = prefix + f'norwr'
        elif sensitivity_test:
            configuration_tag = prefix + f'step{max_steps}'
        else:
            configuration_tag = prefix + f'{max_neigh}'

        output_dir = f"rwr_results/{configuration_tag}"

        adj_list = dict()  # IN  adj_list['p123'] = ['u456', 'n789', ...]

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        print("\n" + "- " * 10 + configuration_tag + " -" * 10 + "\n")
        print('Files output to', output_dir)
        random_walk_with_restart()
