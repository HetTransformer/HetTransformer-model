import os
import json
from typing import List
import torch
from tqdm import tqdm
from text_embedder import TextEmbedder
from multiprocessing import Manager, Process, Pool
from os import listdir
from os.path import join

# input
in_dir = 'data/PHEME'
involved_dir = f'data/rwr_results/pheme_n5_p5_u100'

# output
out_dir = in_dir + '/text_embeddings'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

configs = {
    'tweet' : {
        'model name' : 'vinai/bertweet-base',
        'batch size' : 256,  # X: 1024, 512, O: 32, 64, 128, 256
    },
}
device = 'cuda:0'

def save_embed_file(dir, tid, feature):
    f = open('{}/{}.txt'.format(dir, tid), 'w')
    f.writelines(' '.join(['{:.8f}'.format(v) for v in feature]) + '\n')
    f.close()

def save_embeddings_worker(dir_name, ids, features):
    for i in tqdm(range(len(ids)), desc='save embed'):
        save_embed_file(os.path.join(out_dir, dir_name), ids[i], features[i])

def embed_text(ids, texts, max_seq_len, config, dir_name, num_process = 8):
    def skip_existed(id_list, text_list, dir_name):
        print('skip existed and empty...')
        dn = os.path.join(out_dir, dir_name)
        existed = set([fn for fn in os.listdir(dn) if os.stat(os.path.join(dn,fn)).st_size > 0]) if not os.path.isdir(dn) else set()
        new_id_list, new_text_list = [], []
        for id, text in zip(tqdm(id_list, desc='skip existed'), text_list):
            if id + '.txt' not in existed and len(text) > 0:
                new_id_list.append(id)
                new_text_list.append(text)
        return new_id_list, new_text_list
    def embed_text_sequential(texts):
        print('embed_text_sequential...')
        dir = os.path.join(out_dir, dir_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        batch_size = config['batch size']
        num_batches = (len(texts) + batch_size - 1) // batch_size
        embedder = TextEmbedder(max_seq_len, config['model name'], device=device)
        features = torch.zeros(len(texts), embedder.embed_dim)
        for i in tqdm(range(num_batches), desc='embed text'):
            mn = i * batch_size
            mx = min(len(texts), (i + 1) * batch_size)
            features[mn:mx] = embedder(texts[mn:mx])[:, 0, :].squeeze(1)
        return features
    def save_embeddings_parallel(ids, features):
        dir = os.path.join(out_dir, dir_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        jobs = []
        per_process = (len(ids) + num_process - 1) // num_process
        for i in range(num_process):
            mn, mx = i * per_process, min((i+1)*per_process, len(ids))
            p = Process(target=save_embeddings_worker, args=(dir_name, ids[mn:mx], features[mn:mx]))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
    ids, texts = skip_existed(ids, texts, dir_name)
    print(f'id len = {len(ids)} after skip_existed')
    if len(ids) == 0:
        return
    features = embed_text_sequential(texts)
    save_embeddings_parallel(ids, features)

def process_tweets():
    print('process tweets')
    with open(os.path.join(involved_dir, 'n_involved.txt'), 'r') as f:
        involved_ids = set(f.readlines()[0].strip().split(' '))
    with open(os.path.join(involved_dir, 'p_involved.txt'), 'r') as f:
        involved_ids = involved_ids.union(set(f.readlines()[0].strip().split(' ')))
    input = {k : [] for k in ['ids', 'text']}
    for event_raw in listdir(in_dir):
        if event_raw[-16:] != '-all-rnr-threads': continue
        # {event}-all-rnr-threads
        event = event_raw[:-16]
        for rumority in ['non-rumours', 'rumours']:
            for news_id in tqdm(listdir(join(in_dir, event_raw, rumority)), desc=f'{event}-{rumority}'):
                if news_id == '.DS_Store': continue
                for folder in ['source-tweets', 'reactions']:
                    tweets_dir = join(in_dir, event_raw, rumority, news_id, folder)
                    for tweets_fn in listdir(tweets_dir):
                        if tweets_fn == '.DS_Store': continue
                        if 'p' + tweets_fn.split('.')[0] not in involved_ids and \
                            'n' + tweets_fn.split('.')[0] not in involved_ids:
                            continue
                        with open(os.path.join(tweets_dir, tweets_fn), 'r') as f:
                            tweet = json.load(f)
                            input['ids'].append(tweet["id_str"])
                            input['text'].append(tweet['text'])
    embed_text(input['ids'], input['text'], max_seq_len=49, config=configs['tweet'], dir_name='tweet_text')

def process_user_worker(news_root, news_id, involved_uids, return_list, i, total):
    id_desc = []
    if news_id == '.DS_Store': 
        return
    for folder in ['source-tweets', 'reactions']:
        tweets_dir = join(news_root, news_id, folder)
        for tweets_fn in listdir(tweets_dir):
            if tweets_fn == '.DS_Store': continue
            with open(join(tweets_dir, tweets_fn), 'r') as f:
                tweet = json.load(f)
                if 'u' + tweet['user']["id_str"] in involved_uids and tweet['user']['description'] != None:
                    id_desc.append((tweet['user']["id_str"], tweet['user']['description']))
    return_list.append(id_desc)
    if i % 20 == 0:
        print('process_user_worker {:7}/{:7} {:.5}'.format(i, total, i/total))

def process_user_description(num_process = 8):
    print('process_user_description')
    manager = Manager()
    with open(os.path.join(involved_dir, 'u_involved.txt'), 'r') as f:
        involved_uids = set(f.readlines()[0].strip().split(' '))
    for event_raw in listdir(in_dir):
        if event_raw[-16:] != '-all-rnr-threads': continue
        for rumority in ['non-rumours', 'rumours']:
            news_root = join(in_dir, event_raw, rumority)
            dir_list = listdir(news_root)
            return_list = manager.list()
            with Pool(num_process) as p:
                p.starmap(process_user_worker, [(news_root, dir_list[i], involved_uids, return_list, i, len(dir_list)) for i in range(len(dir_list))])
            ids = [i for e in return_list for (i, j) in e]
            description = [j for e in return_list for (i, j) in e]
            embed_text(ids, description, max_seq_len=49, config=configs['tweet'], dir_name='user_description', num_process=num_process)

if __name__ == '__main__':
    process_tweets()
    process_user_description()