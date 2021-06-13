"""
NOTE All IDs are string.
"""
from itertools import product
from tqdm import tqdm
from collections import defaultdict
from os import listdir
from os.path import join
from json import load

in_dir = 'pheme-figshare'
out_dir = 'pheme-figshare'
node_files = {
}
edge_files = {
    ('n', 'n'): 'PhemeNewsNews.txt',
    ('n', 'p'): 'PhemeNewsPost.txt',
    ('n', 'u'): 'PhemeNewsUser.txt',
    ('p', 'p'): 'PhemePostPost.txt',
    ('p', 'u'): 'PhemePostUser.txt',
    ('u', 'u'): 'PhemeUserUser.txt',
}

def process():
    
    def write_edge(d, fn):
        with open(join(out_dir, fn), 'w') as f:
            for (s, t), v in d.items():
                f.write(f'{s}\t{t}\t{v}\n')
    
    def write_edges_from_structure(root, tree, level):
        # An empty tree is a zero-len list.
        if len(tree) == 0:
            return
        for child, subtree in tree.items():
            if level == 0:
                edges[('n', 'p')][(root, child)] += 1
            else:
                edges[('p', 'p')][(root, child)] += 1
                edges[('p', 'p')][(child, root)] += 1
            write_edges_from_structure(child, subtree, level + 1)
        
    def write_user_edges(folder, tweet_fname, tweet_type):
        tweet = load(open(join(news_root, folder, tweet_fname), 'r'))
        tweet_id, user_id = tweet["id_str"], tweet["user"]["id_str"]
        # n-u or p-u
        edges[(tweet_type, 'u')][tweet_id, user_id] += 1
        # u-u
        another_user_id = tweet["in_reply_to_user_id_str"]
        if another_user_id != None:
            edges[('u', 'u')][user_id, another_user_id] += 1
            edges[('u', 'u')][another_user_id, user_id] += 1

    edges = {k : defaultdict(int) for k in edge_files.keys()}

    for event_raw in listdir(in_dir):
        if not event_raw.endswith('-all-rnr-threads'): continue
        # {event}-all-rnr-threads
        event = event_raw[:-16]
        same_event_news = set()
        for rumority in ['non-rumours', 'rumours']:
            for news_id in tqdm(listdir(join(in_dir, event_raw, rumority)), desc=f'{event}-{rumority}'):
                if news_id == '.DS_Store': continue
                same_event_news.add(news_id)
                news_root = join(in_dir, event_raw, rumority, news_id)
                # n-p, p-p
                structure = load(open(join(news_root, 'structure.json'), 'r'))
                write_edges_from_structure(news_id, structure[news_id], 0)
                # n-u
                write_user_edges('source-tweets', f'{news_id}.json', 'n')
                # p-u, u-u
                for tweet_file_name in listdir(join(news_root, 'reactions')):
                    if tweet_file_name == '.DS_Store': continue
                    write_user_edges('reactions', tweet_file_name, 'p')
        for news_id_1 in same_event_news:
            for news_id_2 in same_event_news:
                edges[('n', 'n')][news_id_1, news_id_2] += 1

    for k, v in edge_files.items():
        write_edge(edges[k], v)

if __name__ == '__main__':
    process()