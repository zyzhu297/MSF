import json
import os
import sys
sys.path.append('/home/zyzhu/MORE') 
from tqdm import tqdm
import jsonlines
import torch
import numpy as np
import clip as clip_folder
import argparse
import shutil
from PIL import Image
import lmdb
import pickle
from src.model.blip2_qformer import Blip2Qformer
import logging
import re

def get_files(file_dir):
#遍历filepath下所有文件，忽略子目录
    all_file = []
    files = os.listdir(file_dir)
    for fi in files:
        fi_d = os.path.join(file_dir,fi)            
        if os.path.isdir(fi_d):
            continue             
        else:
            all_file.append(fi_d)
            
    return all_file

# def get_data(data_dir="datas/commongen", splits=["train", "dev", "test"]):
#     all_data = []
#     for split in splits:
#         fname = os.path.join(data_dir, "commongen.{}.src_alpha.txt".format(split))
#         with open(fname, "r") as f:
#             datas = f.readlines()
#         for i, data in enumerate(datas):
#             d = {}
#             data = data.strip()
#             d["id"] = "{}_{}".format(split, i)
#             d["sent"] = ", ".join(data.split(" "))
#             all_data.append(d)
#     return all_data

def get_feature_from_bing(data_dir, data_name, save_name, debug_mode=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = os.path.join(data_dir, data_name)
    save_path = os.path.join(data_dir, save_name)
    model = Blip2Qformer(
        vision_path="src/pre_trained_lm/eva_vit_g.pth",
        former_path="src/pre_trained_lm/bert-base-uncased",
    )
    msg = model.load_from_pretrained("src/pre_trained_lm/blip2_pretrained.pth")
    logging.info(msg)

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    model = model.to(device)
    model.eval()

    txt_processors = model.get_text_processor()

    env = lmdb.open(save_path, readonly=False, create=True, map_size=4 * 1024**4)
    txn = env.begin(write=True)

    with open(os.path.join(data_dir, "bing_text_for_{}.json".format(data_name)), "r") as f:
        bing_text = json.load(f)
    if debug_mode:
        bing_text = bing_text[:20]

    with torch.no_grad():
        cnt = 0
        for d in tqdm(bing_text.values()):
            concepts = d["concepts"]
            texts = d["bing_results"]
            for i, text in enumerate(texts):
                key = "{}_{}".format(concepts, i).encode()
                c = txn.get(key)
                if c is not None:
                    pass
                else:
                    title = re.sub(r'http\S+|\S+.com', '', text["Title"])
                    description = re.sub(r'http\S+|\S+.com', '', text["Description"])
                    if i == 0:
                        text = "content is {}".format(description)
                    else:
                        text = "title is {}, content is {}".format(title, description)
                    text_input = txt_processors(text)
                    sample = {"text_input": [text_input]}
                    text_feature = model.extract_features(sample, mode="text").text_embeds.cpu()

                    inputs = {"text_feature": text_feature}
                    value = pickle.dumps(inputs)

                    txn.put(key, value)
                    cnt += 1
                    if cnt % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
        txn.commit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="./datas/", type=str)
    parser.add_argument(
        "--data_name", default="MVSA_Single", type=str)
    parser.add_argument(
        "--save_name", default="blip2_text_feats.lmdb", type=str)
    parser.add_argument(
        "--debug_mode",  action='store_true')

    args = parser.parse_args()

    get_feature_from_bing(args.data_dir, args.data_name, args.save_name, debug_mode=args.debug_mode)