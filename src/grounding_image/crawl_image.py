import json
import os
import sys 
from tqdm import tqdm
import jsonlines
import torch
import numpy as np
import clip
import argparse
import shutil
from PIL import Image

def load_jsonl(fname):
    datas = []
    with open(fname, "r") as f:
        for item in jsonlines.Reader(f):
            datas.append(item)
            
    return datas

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def get_data(data_dir="datas", data_name='few_shot', is_few=True, few='1', splits=["train", "dev", "test"]):
    all_data = []
    # data_dir = os.path.join(data_dir, data_name)
    for split in splits:
        if split != 'test' and is_few:
            split = split + '_few' + few
        fname = os.path.join(data_dir, f"{data_name}_{split}.json")
        datas = load_json(fname)
        # print(datas)
        for i, data in datas.items():
            d = {}
            d["id"] = "{}_{}".format(i, split)
            d["sent"] = data['concepts'].strip() 
            d["image"] = data["image"]
            all_data.append(d)
    return all_data


def ground_from_bing(all_data, data_dir="./datas/", data_name='few_shot', max_num=10, prompt="photo of \"{}\""):
    print("ground datas from bing retrieval")
    from icrawler.builtin import BingImageCrawler

    image_dir_name = "bing_image_for_{}".format(data_name)
    # data_dir = os.path.join(data_dir, data_name)
    save_data = {}
    for d in tqdm(all_data):
        qid = d["id"]
        # if "commongen" in data_dir:
        dir_name = d["sent"].split(", ")
        dir_name = sorted(dir_name)
        dir_name = "#".join(dir_name)
        # else:
            # dir_name = qid
        sent = d["sent"]
        sent = prompt.format(sent)
        image_dir = os.path.join(data_dir, image_dir_name, dir_name)
        os.makedirs(image_dir, exist_ok=True)

        filenames = os.listdir(image_dir)

        if len(filenames) < max_num / 2:
            shutil.rmtree(image_dir)
            os.makedirs(image_dir, exist_ok=True)

            bing_storage = {'root_dir': image_dir}
            bing_crawler = BingImageCrawler(downloader_threads=8, storage=bing_storage)
            bing_crawler.crawl(keyword=sent.strip(), max_num=max_num+5)

        filenames = os.listdir(image_dir)
        save_data[qid] = [d["image"]]
        for fn in filenames:
            save_data[qid].append(os.path.join(data_dir, image_dir_name, dir_name, fn))
        # print(save_data)
    save_dir = os.path.join(data_dir, "bing_image_for_{}.json".format(data_name))
    with open(save_dir, "w") as f:
        json.dump(save_data, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--is_few", default=False, type=bool)
    parser.add_argument(
        "--few", default='1', type=str)
    parser.add_argument(
        "--data_name", default="twitter2015", type=str)
    parser.add_argument(
        "--grounding_source", default="dataset", type=str)
    parser.add_argument(
        "--data_dir", default="./datas/IJCAI2019_data", type=str)
    parser.add_argument(
        "--max_image_num", default=20, type=int)
    parser.add_argument(
        "--prompt", default="a photo of {}", type=str)
    parser.add_argument(
        "--debug_mode",  action='store_true')

    args = parser.parse_args()

    # load data
    all_data = get_data(args.data_dir, args.data_name, args.is_few, args.few)
    # print(all_data)
    if args.debug_mode:
        all_data = all_data[:5]

    ground_from_bing(all_data, data_dir=args.data_dir, data_name=args.data_name, max_num=args.max_image_num, prompt=args.prompt)

    # get_few_data()