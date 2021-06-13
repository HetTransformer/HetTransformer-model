import os
import json
from tqdm import tqdm

prefix = 'FakeNewsNet/code/'
ds_path = prefix + 'fakenewsnet_dataset'
dataset = ['politifact', 'gossipcop']
subset = ['fake', 'real']

downloaded_image_dir = 'FakeNewsNet/top_images'
err_res_path = 'FakeNewsNet/img_err_res_news_id.log'

keys = [
    '# News articles', '# News articles with text',
    '# News articles with image', '# News with images downloaded', '# News with images processed',
    '# Users posting tweets', 
    '# Tweets posting news', 
    '# Tweets with retweets',
    '# Followers', '# Followees', 'Average # followers', 'Average # followees', 'Average # friends',
    '# News with retweets', '# News with retweets downloaded', '# News w non-emp retweets downloaded', 
    '# News with tweets', '# News with tweets downloaded',
    '# Retweets', '# Retweets downloaded',
]

uids = set()
tweets_with_retweets = set()
num_followers, num_friends = [], []

all_stats = {k1 : {k2 : {} for k2 in subset} for k1 in dataset}

downloaded_images = os.listdir(downloaded_image_dir)
downloaded_images = [e.split('.')[0] for e in downloaded_images]

with open(err_res_path, 'r') as f:
    error_images = set([e.strip() for e in f.readlines()])

def update_user_stat(user):
    uids.add(user['id'])
    num_followers.append(user['followers_count'])
    num_friends.append(user['friends_count'])

for ds in dataset:
    for ss in subset:
        print(ds, ss, 'starts')
        stats = {key : 0 for key in keys}
        uids = set()
        num_followers, num_friends = [], []
        img_processed_nid = set()
        news_list = os.listdir(os.path.join(ds_path, ds, ss))
        stats['# News articles'] += len(news_list)
        for nid in tqdm(news_list, desc=ds+' '+ss):
            # if nid in downloaded_images:
            #     img_processed_nid.add(nid)
            #     stats['# News images downloaded'] += 1
            # elif nid in error_images:
            #     img_processed_nid.add(nid)
            news_content_path = os.path.join(ds_path, ds, ss, nid, 'news content.json')
            tweet_path = os.path.join(ds_path, ds, ss, nid, 'tweets')
            retweet_path = os.path.join(ds_path, ds, ss, nid, 'retweets')
            has_content, has_tweets, has_retweets = False, False, False
            # if os.path.isfile(news_content_path):
            #     with open(news_content_path, 'r') as fin:
            #         # has_content = True
            #         news_content = json.load(fin)
            #         stats['# News articles with text'] += 1 if len(news_content['text']) > 0 else 0
            #         stats['# News articles with image'] += 1 if len(news_content['top_img']) > 0 else 0
            if os.path.isdir(tweet_path):
                has_tweets = True
                tweet_list = os.listdir(tweet_path)
                stats['# News with tweets downloaded'] += 1
                stats['# Tweets posting news'] += len(tweet_list)
                # for tweet_f in tweet_list:
                #     with open(os.path.join(tweet_path, tweet_f), 'r') as fin:
                #         tweet = json.load(fin)
                #     update_user_stat(tweet['user'])
                #     if tweet['retweet_count'] > 0:
                #         stats['# Tweets with retweets'] += 1
                #         stats['# Retweets'] += tweet['retweet_count']
                #         has_retweets = True
                if has_retweets:
                    stats['# News with retweets'] += 1
            has_retweets, has_ne_retweets = False, False
            if os.path.isdir(retweet_path):
                retweet_list = os.listdir(retweet_path)
                for retweet_f in retweet_list:
                    fsz = os.stat(os.path.join(retweet_path, retweet_f)).st_size
                    has_retweets = True
                    if fsz > 16:  # empty retweet: 16 bytes
                        has_ne_retweets = True
                    # with open(os.path.join(retweet_path, retweet_f), 'r') as fin:
                    #     retweets = json.load(fin)['retweets']
                    # if len(retweets) > 0:
                    #     has_retweets = True
                    #     stats['# Retweets downloaded'] += len(retweets)
                    #     [update_user_stat(retweet['user']) for retweet in retweets]
                    #     tweets_with_retweets.add(retweet_f.split('.')[0])
                if has_retweets:
                    stats['# News with retweets downloaded'] += 1
                if has_ne_retweets:
                    stats['# News w non-emp retweets downloaded'] += 1
        stats['# Users posting tweets'] = len(uids)
        stats['# Tweets with retweets'] = len(tweets_with_retweets)
        stats['Average # followers'] = sum(num_followers) / len(num_followers) if len(num_followers) > 0 else -1
        stats['Average # friends'] = sum(num_friends) / len(num_friends) if len(num_friends) > 0 else -1
        stats['# News images processed'] = len(img_processed_nid)
        print(ds, ss, 'ends')
        all_stats[ds][ss] = stats

print('#' * 10 + ' Overall Stats ' + '#' * 10)

print('\\toprule')

print(' ' * 40, end = "")
for ds in dataset:
    print(' & {:10} & {:10}'.format(ds, ds), end='')


print('\\\\')
print(' ' * 40, end = "")
for i in range(2):
    print(' & {:10} & {:10}'.format('Fake', 'Real'), end='')

    
print('\\\\')

print('\\midrule')

for k in keys:
    print('{:40}'.format(k.replace('#', "\\#")), end='')
    for ds in dataset:
        for ss in subset:
            print(' & {:10}'.format(int(all_stats[ds][ss][k])), end='')
    print('\\\\')

print('\\bottomrule')
    