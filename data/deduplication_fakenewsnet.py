"""
Error code explained: https://developer.twitter.com/en/support/twitter-api/error-troubleshooting
"""
import os
import json
import pprint
pp = pprint.PrettyPrinter(indent=4)

in_dir = 'data/FakeNewsNet/code'
out_f = 'data/FakeNewsNet/retweet_logged_ids.json'

again_key = 'again (re)tweet ids'
not_again_key = 'not again (re)tweet ids'

dls = os.listdir(in_dir)
dls = [e for e in dls if e.startswith('data_collection_') and e.endswith('.log')]

again = {
}
not_again = {
    "twython.exceptions.TwythonError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))" : 0,
    'twython.exceptions.TwythonError: Twitter API returned a 500 (Internal Server Error), Internal error' : 0,
    "twython.exceptions.TwythonError: Twitter API returned a 503" : 0,
    
    "twython.exceptions.TwythonError: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url:" : 0,
    "twython.exceptions.TwythonError: HTTPSConnectionPool(host='api.twitter.com', port=443): Read timed out. (read timeout=" : 0,
    'twython.exceptions.TwythonError: Twitter API returned a 403 (Forbidden), Forbidden.' : 0,
    'twython.exceptions.TwythonError: Twitter API returned a 404 (Not Found), Sorry, that page does not exist.' : 0,
}

again_tids, not_again_tids = [], []
for dl in dls:
    with open(os.path.join(in_dir, dl), 'r') as f:
        lines = f.readlines()  # 960,997 lines
    ptr = 0
    while ptr < len(lines):
        segs = lines[ptr].split(' ')
        if len(segs) < 4 or segs[3] != 'retweet_collection':
            ptr += 1
            continue
        if segs[5] == 'Exception':
            # 2020-10-16 07:22:35,113 70678 retweet_collection ERROR Exception in getting retweets for tweet id 1019659493041373184 using connection <Twython: uNvRntDxircQommDokrEABqPR>
            tid = segs[12]
        elif segs[5] == 'Twython':
            # 2021-03-06 05:39:41,266 51796 retweet_collection ERROR Twython API rate limit exception - tweet id : 1061010976525303808
            tid = segs[14].strip()
        try:
            assert tid.isdigit()
        except Exception as e:
            print('except Exception as e:', lines[ptr], e.__repr__())
        while ptr < len(lines) and not lines[ptr].startswith('twython.exceptions.TwythonError:'):
            ptr += 1
        if ptr == len(lines):
            break
        found = False
        for msg in again.keys():
            if lines[ptr].startswith(msg):
                again_tids.append(tid)
                again[msg] += 1
                found = True
                break
        if not found:
            for msg in not_again.keys():
                if lines[ptr].startswith(msg):
                    not_again_tids.append(tid)
                    not_again[msg] += 1
                    found = True
                    break
        if not found:
            print('if not found:', tid, lines[ptr])
        ptr += 1


pp.pprint(again)
pp.pprint(not_again)

again_tids = set(again_tids)
not_again_tids = set(not_again_tids)
for na in not_again_tids:
    if na in again_tids:
        again_tids.remove(na)


union = again_tids.union(not_again_tids)
assert (len(again_tids) + len(not_again_tids)) == len(union)  # should be disjoint

again_tids = list(again_tids)
not_again_tids = list(not_again_tids)
again_tids.sort()
not_again_tids.sort()

content = {
    again_key : again_tids,
    not_again_key : not_again_tids
}
with open(out_f, 'w') as f:
    f.write(json.dumps(content, indent=4, sort_keys=True))