import json
import os
import urllib.request
import http.cookiejar

data_dirs = ["/data/FakeNewsNet/PolitiFact/Real_News/", "/data/FakeNewsNet/PolitiFact/Fake_News/", "/data/FakeNewsNet/GossipCop/Real_News/", "/data/FakeNewsNet/GossipCop/Fake_News/"]

for data_dir in data_dirs:
    all_json = os.listdir(data_dir)
    k = 0
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-Agent' , 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 ')]
    urllib.request.install_opener(opener)
    errors = []
    picture_error = []
    for j_name in all_json:
        j = open(data_dir+j_name,'rb')
        info = json.load(j)
        try:
            print(info['top_img'])
            try:
                urllib.request.urlretrieve(info['top_img'], data_dir[:-6]+"/img/"+j_name[:-5]+".jpg")
            except:
                print(j_name)
                picture_error.append(j_name)
        except:
            errors.append(j_name)
    print(len(picture_error))
    print(len(errors))
