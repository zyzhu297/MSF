from bs4 import BeautifulSoup
import requests
import re
import os
import argparse
import tqdm
from process_captions import process, load_json
import json
from multiprocessing import Pool
import threading
import re

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

# maybe need to change some values
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'close',
    'cookie': 'MUIDB=03707BE7F8EE6573255C6F40F9C364BC'
        }

def get_bing_url(keywords):
    keywords = keywords.strip('\n')
    bing_url = re.sub(r'^', 'https://cn.bing.com/search?q=', keywords)
    bing_url = re.sub(r'\s', '+', bing_url)
    bing_url += '&setlang=en'
    return bing_url

def clean_description(description):
    # 移除日期（格式示例: "Jul 23, 2015"）
    description = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}\b', '', description)
    # 移除 "Estimated Reading Time: X mins"
    description = re.sub(r'Estimated Reading Time: \d+ mins', '', description)
    return description.strip()

def query_bing(new_key, pages=2):
    l = []
    title_set = set()
    title_idx = []

    bing_url = get_bing_url(new_key)
    for i in range(0,10*pages,10):
        if len(l) >= 20:
            break
        o = {}
        url = bing_url + "&rdr=1&first={}&FORM=PERE&sp=-1&lq=0&pq=&sc=0-0&qs=n&sk=&ghacc=0".format(i+1)
        try:
            content = requests.get(url=url, timeout=30, headers=headers)
            soup = BeautifulSoup(content.text, 'html.parser')

            completeData = soup.find_all("li",{"class":"b_algo"})
            # print(soup)
            for i in range(0, len(completeData)):
                try:
                    o = {}
                    o["Title"] = completeData[i].h2.a.text.strip()
                    # o["link"]=completeData[i].find("a").get("href")
                    o["Description"]=completeData[i].find("div", {"class":"b_caption"}).text.strip()
                    # print(o["Description"])
                    # input()
                    # o["Description"] = o["Description"][2:]
                    if o["Description"] == "":
                        o["Description"]=completeData[i].find("p").text
                    o["Description"] = o["Description"].split("\xa0· ")[-1].strip()

                    if o["Title"] not in title_set:
                        title_set.add(o["Title"])
                        title_idx.append(o["Title"])
                    else:
                        idx = title_idx.index(o["Title"])
                        if l[idx]["Description"] == o["Description"]:
                            continue
                    if not is_contains_chinese(o["Title"] + " " + o["Description"]):
                        o["Title"] = re.sub(r'http\S+|\S+.com', '', o["Title"])
                        o["Description"] = re.sub(r'http\S+|\S+.com', '', o["Description"])
                        o["Description"] = clean_description(o["Description"])
                        l.append(o)
                    # print(o)
                    # print(l)
                    # input()
                    # print(o)
                except:
                    pass
        except:
            pass
        # print(idx, len(l))
        # if idx >= 150:
        #     print(new_key)
        #     input()
        # print(url)
        # print(len(l), l)
        # print('-=-=-'*12)
    return l

# def get_data(data_dir="datas", data_name='few-shot', splits=["train", "dev"]):
#     all_data = []
#     data_dir = os.path.join(data_dir, data_name)
#     for split in splits:
#         fname = os.path.join(data_dir, "{}.json".format(split))
#         with open(fname, "r") as f:
#             datas = f.readlines()
#         for i, data in enumerate(datas):
#             d = {}
#             data = eval(data)
#             d["id"] = "{}_{}".format(split, i)
#             concepts = data['concepts'].strip() 
#             d["sent"] = ", ".join(concepts.split(" "))
#             all_data.append(d)
#     return all_data
def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data

def get_data(data_dir="datas", data_name='few_shot', is_few=True, few='1', splits=["train", "dev", "test"]):
    all_data = []
    # mvsa要还回来
    # data_dir = os.path.join(data_dir, data_name)
    for split in splits:
        if split != 'test' and is_few:
            split = split + '_few' + few
        fname = os.path.join(data_dir, f"{data_name}_{split}.json")
        datas = load_json(fname)
        # print(datas)
        for i, data in datas.items():
            d = {}
            print(data)
            d["id"] = "{}_{}".format(i, split)
            d["sent"] = data['concepts'].strip() 
            # I think it's better without comma.
            d["sent"] = ''.join(d["sent"].split(','))
            d["text"] = data["text"].strip()
            all_data.append(d)
    return all_data


# print(query_bing('Startup Bright.md $3.5 million doctors appointment 90 seconds'))
# input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--is_few", default=False, type=bool)
    parser.add_argument(
        "--few", default='1', type=str)
    parser.add_argument(
        "--data_name", default="twitter2017", type=str)
    parser.add_argument(
        "--grounding_source", default="dataset", type=str)
    parser.add_argument(
        "--data_dir", default="./datas/IJCAI2019_data", type=str)
    parser.add_argument(
        "--save_dir", default="./datas/IJCAI2019_data", type=str)
    # parser.add_argument(
    #     "--data_name", default="MVSA_Single", type=str)
    # parser.add_argument(
    #     "--data_dir", default="./datas/", type=str)
    # parser.add_argument(
    #     "--save_dir", default="./datas/", type=str)
    parser.add_argument(
        "--n_threads", default=1, type=int)
    parser.add_argument(
        "--pages", default=10, type=int)

    args = parser.parse_args()

    all_data = get_data(args.data_dir, args.data_name, args.is_few, args.few)
    # all_data = get_data(args.data_dir, args.data_name)
    print("data loaded")
    # print(all_data)
    querys = set()
    for d in all_data:
        querys.add((d['id'], d["sent"], d["text"]))
    output_data_ = []
    if os.path.exists(os.path.join(args.save_dir, "bing_text_for_{}.json".format(args.data_name))):
        with open(os.path.join(args.save_dir, "bing_text_for_{}.json".format(args.data_name)), "r") as f:
            output_data_ = json.load(f)
    c2i = {}
    # print(output_data_)
    for i, d in enumerate(output_data_):
        print(d)
        if len(output_data_[d]["bing_results"]) > 0:
            c2i[output_data_[d]["concepts"]] = i

    def func(query):
        if query[1] not in c2i:
            res = [{"Title": "Main text", "Description": query[2]}]
            res.extend(query_bing(query[1], pages=args.pages))

            return {
                'qid': query[0],
                "concepts": query[1],
                "bing_results": res,
                }
        else:
            return output_data_[c2i[query[1]]]

    print("searching...")
    output_data = []
    if args.n_threads == 1:
        for query in tqdm.tqdm(querys):
            output_data.append(func(query))
    else:
        with Pool(args.n_threads) as p:
            output_data = list(tqdm.tqdm(p.imap(func, querys), total=len(querys)))
    tmp_data = {}
    for data in output_data:
        tmp_data[data['qid']] = {'concepts':data['concepts'], 'bing_results':data['bing_results']}
    # mvsa要还回来
    # save_dir = os.path.join(args.save_dir, args.data_name)
    output_path = os.path.join(args.save_dir, "bing_text_for_{}.json".format(args.data_name))
    with open(output_path, 'w') as fi:
        json.dump(tmp_data, fi)
        

