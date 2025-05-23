import os
import time
import pickle
import json
import logging
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase 
from transformers import AutoTokenizer
from dataclasses import dataclass
import lmdb
import clip
import random
import spacy    
from .utils import get_image_input, get_graph_input, get_text_input, _torch_collate_batch, load_json

nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorCommongen:
    tokenizer: PreTrainedTokenizerBase
    qformer_tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        return self.torch_call(features)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        input_ids = [e['input_ids'] for e in examples]
        batch = {
            "input_ids": _torch_collate_batch(input_ids, self.tokenizer)
        }
        # print(len(examples))
        # print('-=-'*90)
        # print(examples)
        # print('313'*28)
        # for t in examples[0]:
        #     print(t)
        # print('-='*20)
        # input()
        # batch['data_ids'] = [e['data_id'] for e in examples]
        origin_image_feature = [e['origin_image_feature'] for e in examples]
        batch['origin_image_feature'] = torch.stack(origin_image_feature)

        attention_mask = (batch['input_ids'] != self.tokenizer.pad_token_id)
        batch['attention_mask'] = attention_mask.long()

        batch["labels"] = _torch_collate_batch([e['labels'] for e in examples], self.tokenizer)
        attention_mask = (batch['labels'] != self.tokenizer.pad_token_id)
        batch['label_attention_mask'] = attention_mask.long()

        word_mask = batch["labels"] != self.tokenizer.pad_token_id
        batch["labels"][~word_mask] = -100

        former_text_input_ids = [e['former_text_input_ids'] for e in examples]
        batch["former_text_input_ids"] = _torch_collate_batch(former_text_input_ids, self.qformer_tokenizer)
        attention_mask = (batch['former_text_input_ids'] != self.qformer_tokenizer.pad_token_id)
        batch['former_text_input_masks'] = attention_mask.long()

        ### image input
        if examples[0].get("image_inputs", None) is not None:
            image_embeds = [e['image_inputs'] for e in examples]
            dim = max(len(x.size()) for x in image_embeds) # x: N,32,H
            max_n = max(x.size(0) for x in image_embeds)
            hid = max(x.size(-1) for x in image_embeds)
            # B, N, H
            padded_image_embeds, image_inputs_attns = None, None
            if dim == 2:
                padded_image_embeds = torch.zeros(len(image_embeds), max_n, hid)
                image_inputs_attns = torch.zeros(len(image_embeds), max_n).long()
                for i, image_embed in enumerate(image_embeds):
                    if len(image_embed) > 0:
                        padded_image_embeds[i, :image_embed.size(0), :image_embed.size(1)] = image_embed
                        image_inputs_attns[i, :image_embed.size(0)] = 1
            # B, N, L, H
            elif dim == 3:
                l = max(x.size(1) if len(x.size()) == 3 else 0 for x in image_embeds)
                padded_image_embeds = torch.zeros(len(image_embeds), max_n, l, hid)
                image_inputs_attns = torch.zeros(len(image_embeds), max_n, l).long()
                for i, image_embed in enumerate(image_embeds):
                    if len(image_embed) > 0:
                        padded_image_embeds[i, :image_embed.size(0), :image_embed.size(1), :image_embed.size(2)] = image_embed
                        image_inputs_attns[i, :image_embed.size(0), :image_embed.size(1)] = 1

            batch["image_inputs"] = padded_image_embeds
            batch["image_inputs_attention_mask"] = image_inputs_attns

        # if examples[0].get("image_qpp", None) is not None:
        #     image_qpp = [e['image_qpp'] for e in examples]
        #     image_qpp = torch.stack(image_qpp) # (B,nxl,d)
        #     batch["image_qpp"] = image_qpp

        if examples[0].get("text_inputs", None) is not None:
            text_inputs = [e['text_inputs'] for e in examples]

            dim = max(len(x.size()) for x in text_inputs) # x: L,H
            max_l = max(x.size(-2) if len(x) != 0 else 0 for x in text_inputs)
            hid = max(x.size(-1) if len(x) != 0 else 0 for x in text_inputs)
            padded_text_embeds, text_inputs_attns = None, None
            # B, L, H
            if dim == 2:
                padded_text_embeds = torch.zeros(len(text_inputs), max_l, hid)
                text_inputs_attns = torch.zeros(len(text_inputs), max_l).long()
                for i, text_embed in enumerate(text_inputs):
                    if len(text_embed) > 0:
                        padded_text_embeds[i, :text_embed.size(0)] = text_embed
                        text_inputs_attns[i, :text_embed.size(0)] = 1
            
            batch['text_inputs'] = padded_text_embeds
            batch['text_inputs_attention_mask'] = text_inputs_attns

        if examples[0].get("text_input_ids", None) is not None:
            text_input_ids = _torch_collate_batch([e['text_input_ids'] for e in examples], self.tokenizer)
            batch['text_input_ids'] = text_input_ids

            attention_mask = (batch['text_input_ids'] != self.tokenizer.pad_token_id)
            batch['text_inputs_attention_mask'] = attention_mask.long()
            
        # print('NJUOIVKJLM', batch['labels'])
        # print('JIOJHUBHJN', batch['label_attention_mask'])
        # print(batch)
        # print(batch['input_ids'])
        # for b, v in batch.items():
        #     print(b, v)
        #     import time
        #     time.sleep(0.2)
        # print('-=-==-' * 20)
        
        return batch

class Lmdb():
    def __init__(self, lmdb_file, readonly=True):
        if readonly:
            self.env = lmdb.open(lmdb_file, lock=False,
                                 readonly=True, create=False)
            self.txn = self.env.begin(buffers=True)
        else:
            # prepro
            self.env = lmdb.open(lmdb_file, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)

    def get(self, image_id):
        key = str(image_id).encode()
        c = self.txn.get(key)
        if not c:
            return None
        image_info = pickle.loads(c)
        return image_info

    def exist(self, image_id):
        key = str(image_id).encode()
        c = self.txn.get(key)
        if not c:
            return False
        else:
            return True

    def get_keys(self):
        keys = []
        for k,v in self.txn.cursor():
            keys.append(bytes(k).decode("utf-8"))
        return keys

    def __del__(self):
        self.env.close()


class CommongenDataset(Dataset):
    def __init__(
        self, 
        args,
        tokenizer,
        qformer_tokenizer,
        split="train",
        graph_obj=None,
        unlabeled=None,
        ):
        
        self.args = args
        self.tokenizer = tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        self.k = args.image_num
        self.split = split
        self.task_prompt = args.task_prompt
        self.unlabeled = unlabeled

        if split in ["val", "dev"]:
            split = "dev"

        # if args.use_text and args.text_grounding_path is not None:
        #     self.text_tokenizer = AutoTokenizer.from_pretrained(
        #         "./pre_trained_lm/bert-base-uncased"
        #     )

        cached_inputs_file = os.path.join(
            args.data_dir,
            "cached",
            "{}.pkl".format(split)
        )

        os.makedirs(os.path.join(
            args.data_dir,
            "cached",
        ), exist_ok=True)

        start = time.time()
        if os.path.exists(cached_inputs_file) and not args.overwrite_cache:
            self.examples = pickle.load(open(cached_inputs_file, 'rb', buffering=4096))
            logger.info(
                f"Loading examples from cached file {cached_inputs_file} [took %.3f s]", time.time() - start
            )

            if args.debug_mode:
                self.examples = self.examples[:100]

        else:
            logger.info(f"Creating examples from dataset file at {args.data_dir}")
            if split == 'dev' and args.is_few:
                split = split + '_few'
            if split == 'train' and args.is_few:
                split = split + '_few' + args.few
            if self.unlabeled is not None:
                if self.unlabeled:
                    split = split + '_unlabeled'
            input_file = os.path.join(args.data_dir, f"{args.data_name}_{split}.json")
            # input_file = os.path.join(args.data_dir, "{}.json".format(split))
            # target_file = os.path.join(args.data_dir, "commongen.{}.tgt.txt".format(split))

            # with open(input_file, "r") as f:
            #     input_datas = f.readlines()
            # with open(target_file, "r") as f:
            #     target_datas = f.readlines()
            print('input_path', input_file)
            with open(input_file, 'r') as f:
                input_datas = json.load(f)
            print('len_input', len(input_datas))
            # print(input_datas)
            # input('jie')
            grounding_image = None
            if args.use_image and args.image_grounding_path is not None:
                grounding_image_ = load_json(args.image_grounding_path)
                grounding_image = {}
                for d in grounding_image_.items():
                    grounding_image[d[0]] = d[1]
                del grounding_image_

            grounding_text = None
            if args.use_text and args.text_grounding_path is not None:
                grounding_text_ = load_json(args.text_grounding_path)
                grounding_text = {}
                for d in grounding_text_.items():
                    grounding_text[d[0]] = (d[1]['concepts'], d[1]['bing_results'])
                del grounding_text_
            # print(grounding_text.keys())
            # input('groundinge')
            self.examples = []
            if args.debug_mode:
                input_datas = input_datas[:200]
                target_datas = target_datas[:200]
            # concept_set = set()
            # for i, (input_data, target_data) in tqdm(enumerate(zip(input_datas, target_datas))):
            for i, (key, input_data) in tqdm(enumerate(input_datas.items())):
                # input_data = input_data.strip()
                # target_data = target_data.strip()
                # previous version which only contain few sample
                # qid = "{}_{}".format(key, split)
                state = split.split('_')[0]
                qid = "{}_{}".format(key, state)
                # print(input_data)
                # print(qid)
                # if split != "train":
                #     if input_data["concepts"] in concept_set:
                #         continue
                #     else:
                #         concept_set.add(input_data["concepts"])

                #TODO fill in the prompt and inputs which is the concepts before.
                source = input_data["concepts"] 
                # TODO training target
                target = input_data["label"]
                if qid not in grounding_text:
                    continue
                image_files, text_files = None, None
                if grounding_image is not None:
                    # image_files = sorted(grounding_image.get(str(qid), []))[:args.image_num]
                    # print('grounding_image', grounding_image)
                    image_files = grounding_image.get(str(qid), [])[:]
                    origin_image = image_files[0]
                    image_files = image_files[1:]
                    # print('image_files', image_files)
                    # print('-=-=-=')
                    # input()
                    
                if grounding_text is not None:
                    text_files = grounding_text.get(str(qid), [])[:]
                    # print('text_files', text_files)
                    # print('-=-=-=')
                    # tmp is the index of the text, here using concepts add index which define in the lmdb
                    origin_text = text_files[1][0]['Description']
                    # print('origin_text', origin_text)
                    tmp = []
                    for i, _ in enumerate(text_files[1][1:]):
                        tmp.append(text_files[0] + '_' + str(i))
                former_text_input = self.qformer_tokenizer(input_data["concepts"], max_length=256)["input_ids"]

                res = {
                    "inputs": source,
                    "data_id": qid,
                    "labels": target,
                    "former_text_input_ids": former_text_input,
                    "image_files": image_files,
                    "text_files": tmp,
                    "origin_text": origin_text,
                    "origin_image": origin_image,
                }
                # print(res)
                # input()
                # print('-' * 20)
                self.examples.append(res)

            pickle.dump(self.examples, open(cached_inputs_file, 'wb', buffering=4096))
            logger.info(
                "Build and Save examples into cached file %s, [took %.3f s]", cached_inputs_file, time.time() - start
            )
            del grounding_image
            del grounding_text

        if args.debug_mode:
             self.examples = self.examples[:200]

        self.image_lmdb = None
        self.text_lmdb = None

        if args.rand_ra:
            examples_ = []
            rand_i = np.arange(0, len(self.examples))
            random.shuffle(rand_i)
            for i, e in enumerate(self.examples):
                e_ = e.copy()
                image_files = self.examples[rand_i[i]]["image_files"].copy() if self.examples[rand_i[i]]["image_files"] is not None else None
                e_["image_files"] = image_files
                text_files = self.examples[rand_i[i]]["text_files"].copy() if self.examples[rand_i[i]]["text_files"] is not None else None
                e_["text_files"] = text_files
                examples_.append(e_)
            self.examples = examples_

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        item = self.examples[i].copy()
        # print(item)
        # input()
        # if self.split == "train" and self.args.random_p > 0:
        #     rand = np.array([0, 1])
        #     prob = np.array([1-self.args.random_p, self.args.random_p])
        #     rand = np.random.choice(a=rand, p=prob)
        #     if rand:
        #         r = np.random.randint(0,len(self.examples))
        #         item["text_files"] = self.examples[r]["text_files"].copy() if self.examples[r]["text_files"] is not None else None
        #         item["image_files"] = self.examples[r]["image_files"].copy() if self.examples[r]["image_files"] is not None else None
        #         item["labels"] = "none"

        item_ = {
            # "input_ids": self.tokenizer(inputs, max_length=512,  return_tensors="pt", padding="longest", truncation=True)["input_ids"],
            "data_id": item["data_id"],
            "input_ids": None,
            "labels": self.tokenizer(item["labels"], max_length=64, truncation=True)["input_ids"],
            "former_text_input_ids": item["former_text_input_ids"]
        }

        if self.args.use_text and self.text_lmdb is None and self.args.text_input_path is not None:
            self.text_lmdb = Lmdb(self.args.text_input_path, readonly=True)
        if item["text_files"] is not None and self.args.use_text:
            # origin_text = item["text_files"]
            # print(origin_text)
            # print('-=-==-==--==--==--==--==--==--==--==--==--==--==')
            k = self.args.text_num
            if self.split == "train":
                if self.args.shuffle_ra:
                    random.shuffle(item["text_files"])
            text_files_ = item["text_files"][:k]
            # text_files = [f['Title'] for f in text_files_]
            # scores = [s for _,s in text_files_]
            
            try:
                text_embeds = [self.text_lmdb.get(str(text_file))["text_feature"] for text_file in text_files_]
                # origin_text = text_embeds[0]
                # text_embeds = text_embeds[1:]
                # print(origin_text)
                # print('-==-==--==--==--==--==--==--==--==--==--==--==')
                if len(text_embeds) > 0:
                    item_["text_inputs"] = torch.cat(text_embeds, dim=-2).squeeze(0) # (l, h)
                else:
                    item_["text_inputs"] = torch.Tensor([])
            except:
                text_embeds = [self.text_lmdb.get(str(text_file))["input_ids"] for text_file in text_files_]
                if len(text_embeds) > 0:
                    item_["text_input_ids"] = torch.cat(text_embeds, dim=-1).squeeze(0) # (l)
                else:
                    item_["text_input_ids"] = torch.Tensor([])

            # print("text_inputs", item_["text_inputs"].shape)

        if self.args.use_image and self.image_lmdb is None and self.args.image_input_path is not None:
            self.image_lmdb = Lmdb(self.args.image_input_path, readonly=True)
        # print(self.image_lmdb.get_keys())
        if item["image_files"] is not None and self.args.use_image:
            # print(self.image_lmdb.get_keys())
            k = self.args.image_num
            if self.split == "train":
                if self.args.shuffle_ra:
                    random.shuffle(item["image_files"])
            image_files_ = item["image_files"][:k]
            # image_files = [f for f in image_files_]
            # scores = [s for _,s in image_files_]
            # print(self.image_lmdb.get_keys())
            image_embeds = [self.image_lmdb.get(str(image_file))["image_feature"] for image_file in image_files_]
            # origin_image = image_embeds[0]
            # image_embeds = image_embeds[1:]
            if len(image_embeds) > 0:
                item_["image_inputs"] = torch.cat(image_embeds, dim=0) # (n, h) or (n, l, h)
            else:
                item_["image_inputs"] = torch.Tensor([])

        origin_image_feature = self.image_lmdb.get(str(item['origin_image']))["image_feature"]
        # print(origin_image_feature)
        if len(origin_image_feature) > 0:
            item_["origin_image_feature"] = origin_image_feature
        else:
            item_["origin_image_feature"] = torch.Tensor([])
        # print(item_["origin_image_feature"])
        # print('-=-='*20)
        # print(origin_image_feature)
        # print('image_feather', origin_image_feature.size())
        if self.args.task_prompt is None:
            self.args.task_prompt = '''Task: Classify the sentiment based on text and image.
            Concepts about the text and image: {}.
            Text: {}.
            Image: [IMG].
            Instruction: Given the extracted features from both the text and image, classify the overall sentiment as positive, neutral, or negative.'''
        input = self.args.task_prompt.format(item["inputs"], item['origin_text'])
        item_["input_ids"] = self.tokenizer(input, max_length=512, truncation=True)["input_ids"]
        # print(item_["input_ids"])
        # print(self.tokenizer.decode(item_["input_ids"]))
        # print(item_)
        # print('o00o000oo00o'*20)
        # input()
        # print(item_)
        # for item in item_:
        #     print(item)
        # print('98765'*21)
        return item_
