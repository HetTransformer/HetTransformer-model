import json
import os
import urllib.request
import http.cookiejar
from os import path

politifact_data_dirs = ["data/FakeNewsNet/code/fakenewsnet_dataset/politifact/real/", "data/FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/"]
gossipcop_data_dirs = ["data/FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/", "data/FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/"]
politifact_img_dir = "data/processed_data/FakeNewsNet/PolitiFact/img/"
gossipcop_img_dir = "data/processed_data/FakeNewsNet/GossipCop/img/"
if not path.exists(politifact_img_dir):
    os.mkdir(politifact_img_dir)
if not path.exists(gossipcop_img_dir):
    os.mkdir(gossipcop_img_dir)
for data_dir in politifact_data_dirs: 
    all_folder = os.listdir(data_dir)
    for folder_name in all_folder:
        folder_dir = data_dir+folder_name
        all_files = os.listdir((folder_dir))
        flag = False
        for file in all_files:
            if file[-4:]=='json':
                j_name = file
                j = open(folder_dir+'/'+file,'rb')
                info = json.load(j)
                flag = True  
        if flag == False:
            continue
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        opener.addheaders = [('User-Agent' , 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 ')]
        urllib.request.install_opener(opener)
        errors = []
        picture_error = []
        try:
            print(info['top_img'])
            try:
                urllib.request.urlretrieve(info['top_img'], politifact_img_dir+j_name[:-5]+".jpg")
            except:
                print(j_name)
                picture_error.append(j_name)
        except:
            errors.append(j_name)
            
for data_dir in gossipcop_data_dirs: 
    all_folder = os.listdir(data_dir)
    for folder_name in all_folder:
        folder_dir = data_dir+folder_name
        all_files = os.listdir((folder_dir))
        flag = False
        for file in all_files:
            if file[-4:]=='json':
                j_name = file
                j = open(folder_dir+'/'+file,'rb')
                info = json.load(j)
                flag = True  
        if flag == False:
            continue
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        opener.addheaders = [('User-Agent' , 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 ')]
        urllib.request.install_opener(opener)
        errors = []
        picture_error = []
        try:
            print(info['top_img'])
            try:
                urllib.request.urlretrieve(info['top_img'], gossipcop_img_dir+j_name[:-5]+".jpg")
            except:
                print(j_name)
                picture_error.append(j_name)
        except:
            errors.append(j_name)

