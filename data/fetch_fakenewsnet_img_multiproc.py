import os
import json
from multiprocessing import Process, Manager
from typing import List, Set
from random import shuffle
from tqdm import tqdm
import requests

n_pro = 8
batch_sz = 128
in_dir = 'FakeNewsNet/code/fakenewsnet_dataset'
out_dir = 'FakeNewsNet/top_images'
err_res_path = 'FakeNewsNet/img_err_res_news_id.log'
sources = ['gossipcop', 'politifact']
labels = ['fake', 'real']

img_file_exts = ["JPG", "PNG", "GIF", "WEBP", "TIFF", "PSD",
                 "RAW", "BMP", "HEIF", "JPEG", "SVG", "AI", "EPS", "PDF"]


def worker(pathes: List[str], process_idx: int, err_res_ids: Set[str], return_dict):
    new_err_res_ids = set()
    for path in pathes:
        content_path = os.path.join(path, 'news content.json')
        news_id = content_path.split(os.path.sep)[-2]
        if not os.path.isfile(content_path):
            # print(f'DEBUG no_content {news_id}')
            new_err_res_ids.add(news_id)
            continue

        if news_id in err_res_ids:
            # print(f'DEBUG in_err_res_ids {news_id}')
            new_err_res_ids.add(news_id)
            continue

        with open(content_path, 'r') as fin:
            news_content = json.load(fin)
        url = news_content['top_img']

        if url == '':
            # print(f'DEBUG empty_url {news_id}')
            new_err_res_ids.add(news_id)
            continue

        file_ext = 'UNK_EXT'
        for ext in img_file_exts:
            ext_idx_upper = url.rfind(ext)
            ext_idx_lower = url.rfind(ext.lower())
            if ext_idx_upper != -1 or ext_idx_lower != -1:
                file_ext = ext
                break
        out_path = os.path.join(out_dir, f'{news_id}.{file_ext}')

        if os.path.isfile(out_path):  # and os.path.getsize(out_path) > 0:
            # print(f'DEBUG processed {news_id}')
            # new_err_res_ids.add(news_id)  # note cuz I did this, error res may simply be sth processed
            continue

        try:
            response = None
            response = requests.get(url, timeout=60)
            if response and response.status_code // 100 == 2:  # 200-299: successful responses
                file = open(out_path, "wb")
                file.write(response.content)
                file.close()
            else:
                # print(f'DEBUG response {news_id} {response.status_code}')
                new_err_res_ids.add(news_id)
        except Exception as e:
            # print(f'ERROR unknown {news_id} {url} {repr(e)}')
            new_err_res_ids.add(news_id)

    return_dict[process_idx] = new_err_res_ids


def download_images(paths: List[str]):
    with open(err_res_path, 'r') as f:
        err_res_ids = set([e.strip() for e in f.readlines()])

    n_batches = (len(paths) + batch_sz - 1) // batch_sz
    for i_b in tqdm(range(n_batches)):
        sb, eb = i_b * batch_sz, min((i_b + 1) * batch_sz, len(paths))
        batch = paths[sb:eb]

        manager = Manager()
        return_dict = manager.dict()
        jobs = []
        n_files_per_p = (len(batch) + n_pro - 1) // n_pro
        for i in range(n_pro):
            si, ei = i * \
                n_files_per_p, min((i + 1) * n_files_per_p, len(batch))
            jobs.append(Process(target=worker, args=(
                batch[si:ei], i, err_res_ids, return_dict)))
            jobs[-1].start()
        for p in jobs:
            p.join()

        for k, v in return_dict.items():
            err_res_ids = err_res_ids.union(v)
        with open(err_res_path, 'w') as f:
            f.writelines('\n'.join(err_res_ids) + '\n')

    print("Finished downloading images")


if __name__ == '__main__':
    paths = []
    for s in sources:
        for l in labels:
            prefix = os.path.join(in_dir, s, l)
            pathes = os.listdir(prefix)
            paths.extend([os.path.join(prefix, path) for path in pathes])

    shuffle(paths)
    download_images(paths)
