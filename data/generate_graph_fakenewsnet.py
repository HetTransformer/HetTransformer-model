"""
NOTE all IDs are string
"""
import json
import os
from tqdm import tqdm
from multiprocessing import Manager, Pool

in_dir = 'data/FakeNewsNet/code/fakenewsnet_dataset'
out_dir = 'data/FakeNewsNet/graph_def'
datasets = ['gossipcop', 'politifact']
subsets = ['fake', 'real']
num_process = 4

def process_worker(ds_dir, news_id, return_list, i, news_ids_len):
    print('process_worker {:7} {:7} {:.4f}'.format(i, news_ids_len, i / news_ids_len))
    np_edges, pu_edges, uu_edges = [], [], []
    source_news, author_news = dict(), dict()
    content_f = os.path.join(ds_dir, news_id, 'news content.json')
    if os.path.isfile(content_f):
        try:
            with open(content_f, 'r') as f:
                content = json.load(f)
            if content["source"] not in source_news.keys():
                source_news[content["source"]] = []
            source_news[content["source"]].append(news_id)
            for author in content["authors"]:
                if len(author.split()) >= 5:  # 'Please Enter Your Name Here'
                    break
                author = author.replace('About ', '')  # 'About Janet Farrow'
                if author not in author_news.keys():
                    author_news[author] = []
                author_news[author].append(news_id)
        except Exception as e:
            print('Error reading news content:', e.__repr__(), content_f)
    tweet_dir = os.path.join(ds_dir, news_id, 'tweets')
    if os.path.isdir(tweet_dir):
        for fname in os.listdir(tweet_dir):
            tweet_id = fname.split('.')[0]
            try:
                with open(os.path.join(tweet_dir, fname), 'r') as f:
                    tweet = json.load(f)
                np_edges.append((news_id, tweet_id))
                pu_edges.append((tweet_id, tweet['user']['id_str']))
            except Exception as e:
                print('Error reading tweet:', e.__repr__(), os.path.join(tweet_dir, fname))
    retweet_dir = os.path.join(ds_dir, news_id, 'retweets')
    if os.path.isdir(retweet_dir):
        for fname in os.listdir(retweet_dir):
            tweet_id = fname.split('.')[0]
            try:
                with open(os.path.join(retweet_dir, fname), 'r') as f:
                    retweets = json.load(f)['retweets']
                for retweet in retweets:
                    forw_user = retweet["user"]["id_str"]
                    orig_user = retweet["retweeted_status"]["user"]["id_str"]
                    uu_edges.append((orig_user, forw_user))
                    uu_edges.append((forw_user, orig_user))
                    pu_edges.append((tweet_id, forw_user))
            except Exception as e:
                print('Error reading retweet:', e.__repr__(), os.path.join(retweet_dir, fname))
    return_list.append((np_edges, pu_edges, uu_edges, source_news, author_news))

def process(ds):
    nn_edges, np_edges, pu_edges, uu_edges = [], [], [], []
    source_news, author_news = dict(), dict()
    for ss in subsets:
        news_ids = os.listdir(os.path.join(in_dir, ds, ss))
        manager = Manager()
        return_list = manager.list()
        ds_dir = os.path.join(in_dir, ds, ss)
        with Pool(num_process) as p:
            p.starmap(process_worker, [(ds_dir, news_ids[i], return_list, i, len(news_ids)) for i in range(len(news_ids))])
        for (_np_edges, _pu_edges, _uu_edges, _source_news, _author_news) in return_list:
            np_edges.extend(_np_edges)
            pu_edges.extend(_pu_edges)
            uu_edges.extend(_uu_edges)
            source_news.update(_source_news)
            author_news.update(_author_news)
    source_news_hist, author_news_hist = dict(), dict()
    for news_id_list in source_news.values():
        if len(news_id_list) not in source_news_hist.keys():
            source_news_hist[len(news_id_list)] = 0
        source_news_hist[len(news_id_list)] += 1
        # print('source news count:', source, len(news_id_list))
        for nid1 in news_id_list:
            for nid2 in news_id_list:
                nn_edges.append((nid1, nid2))
    for news_id_list in author_news.values():
        if len(news_id_list) not in author_news_hist.keys():
            author_news_hist[len(news_id_list)] = 0
        author_news_hist[len(news_id_list)] += 1
        # print('author news count:', author, len(news_id_list))
        for nid1 in news_id_list:
            for nid2 in news_id_list:
                nn_edges.append((nid1, nid2))
    stats = [
        '# News-news edges {:10} / {:10}'.format(len(set(nn_edges)), len(nn_edges)),
        '# News-post edges {:10} / {:10}'.format(len(set(np_edges)), len(np_edges)),
        '# Post-user edges {:10} / {:10}'.format(len(set(pu_edges)), len(pu_edges)),
        '# User-user edges {:10} / {:10}'.format(len(set(uu_edges)), len(uu_edges)),
        'source_news_hist  ' + json.dumps(source_news_hist, indent=4, sort_keys=True),
        'author_news_hist  ' + json.dumps(author_news_hist, indent=4, sort_keys=True),
    ]
    fname_dict = {
        'news-news edges' : nn_edges,
        'news-post edges' : np_edges,
        'post-user edges' : pu_edges,
        'user-user edges' : uu_edges,
        'stats' : [[e,] for e in stats],
    }
    od = os.path.join(out_dir, ds)
    if not os.path.isdir(od):
        os.mkdir(od)
    for k, v in fname_dict.items():
        with open(os.path.join(od, k + '.txt'), 'w') as f:
            f.write('\n'.join([' '.join(e) for e in v]) + '\n')


if __name__ == '__main__':
    for dataset in datasets:
        process(dataset)