import sys
import os
sys.path.append('/home/zyzhu/MORE')
sys.path.append('/home/zyzhu/SemiEvol')
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, AutoTokenizer, T5ForConditionalGeneration
from typing import Any, Dict, List, Optional, Tuple, Union
from model.more_t5 import MoreT5
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
from tqdm import tqdm
import torch
import copy
import json
import random
import datetime
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from data.commongen import CommongenDataset, DataCollatorCommongen
from metric.commongen_metric import commongen_metric_builder
from metric.metric import few_metric_builder
import concurrent.futures
# from common import *
# from utils import *
# from eval import Eval
import threading
from retrying import retry
from func_timeout import func_set_timeout
from data.utils import get_image_input, get_graph_input, get_text_input, _torch_collate_batch, load_json


class ThreadSafeDict:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}

    def get(self, key):
        with self.lock:
            return self.data.get(key)

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

def calculate_entropy(probs):
    prob_list = np.array(probs)
    entropy = - np.sum(prob_list) / len(prob_list)
    return entropy

GLOBAL_RETRIEVAL_CACHE = ThreadSafeDict()

def format_few_shot(data):
    ref_str, img_lst = nr.fewshot(data)
    input_ids = data['input_ids'][0]
    # print(input_ids)
    ques = tokenizer.decode(input_ids)[:-4]
    image = data['origin_image_feature'][0]
    ref_str += "Please answer the following questions.\n"
    ref_str += f"{ques}"
    # print(image.shape)
    img_lst.append(image)
    # system_prompt = FUNCTION_UTILS[question_type]['few_shot_prompt'].format(reference=ref_str)
    # user_prompt = FUNCTION_UTILS[question_type]['format_fn'](data)
    # print(ref_str)
    data['input_ids'] = ref_str
    # print(len(img_lst))
    # print(img_lst)
    data['origin_image_feature'] = torch.stack(img_lst).unsqueeze(0)
    return data

class NearestReference:
    def __init__(self, k=4) -> None:
        self.vectorstore = None
        self.selector = None
        self.k = k

    def read_data(self, data_path):
        df = pd.read_csv(data_path)
        examples = [row.to_dict() for _, row in df.iterrows()]
        return examples

    def embed_data_path(self, data_path, embed_path=None):
        data = self.read_data(data_path)
        return self.embed_data(data, embed_path)

    def embed_data(self, data, embed_path=None):
        embed_path = embed_path or 'tmp/embed'
        
        os.makedirs(embed_path, exist_ok=True)
        # 1. 提取所有 input_ids（假设是 List[torch.Tensor]）
        input_ids_list = [row['input_feature'] for row in data]  # 每个元素是 [1, seq_len, 768]
        
        # 2. 对每个变长 Tensor 做均值池化（变成固定长度向量）
        pooled_embeddings = []
        for tensor in input_ids_list:
            # 去掉 batch 维度（如果有）
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)  # [1, seq_len, 768] → [seq_len, 768]
            # 均值池化（沿序列长度取平均）
            pooled = tensor.mean(dim=0)  # [seq_len, 768] → [768]
            pooled_embeddings.append(pooled)
        
        # 3. 转成 NumPy 并堆叠成 [n, 768]
        embeddings_np = torch.stack(pooled_embeddings).to('cpu').numpy()  # [n, 768]
        
        # 4. 使用 FAISS 存储（不需要文本，用空字符串代替）
        dummy_texts = [""] * len(embeddings_np)
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(dummy_texts, embeddings_np)),
            embedding=embed,  # 仍然需要，但不会被使用
            metadatas=data    # 可选：保留原始数据
        )
        self.vectorstore.save_local(embed_path)

        self.selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore, k=self.k
        )
        return self.vectorstore
        #     embeddings = np.array([row['input_ids'] for row in data])
        
        #     # 2. 创建空的 FAISS 索引并直接添加向量
        #     self.vectorstore = FAISS(embed)  # 仍然需要 embed，但仅用于初始化
        #     self.vectorstore.add_vectors(embeddings, data)  # data 可作为 metadata
            
        #     self.vectorstore.save_local(embed_path)
        # self.selector = SemanticSimilarityExampleSelector(
        #     vectorstore=self.vectorstore, k=self.k
        # )
        # return self.vectorstore

    # @retry
    # @func_set_timeout(2)
    def retrieve(self, question):
        question_str = str(question)  # 转为字符串
        cached_result = GLOBAL_RETRIEVAL_CACHE.get(question_str)
        if cached_result is not None:
            return cached_result
        res = self.vectorstore.similarity_search_by_vector(question, k=self.k)
        # res = self.selector.select_examples({'input_ids': question})
        GLOBAL_RETRIEVAL_CACHE.set(question_str, res)
        return res

    def fewshot(self, question):
        tensor = question['input_feature']
        # 去掉 batch 维度（如果有）
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)  # [1, seq_len, 768] → [seq_len, 768]
        # 均值池化（沿序列长度取平均）
        query_vector = tensor.mean(dim=0)  # [seq_len, 768] → [768]
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.detach().cpu().numpy()  # -> numpy array
            if query_vector.ndim > 1:
                query_vector = query_vector.squeeze()  # 去掉 batch 维 
        # print(query_vector)
        ref = self.retrieve(query_vector)
        # print('ref[0]', ref[0].metadata['data_id'])
        # print(len(ref))
        # print('-=-=-' * 20)
        # input()
        ref_str = 'You are an expert in graphic and textual sentiment analysis. Now, here are several examples of sentiment analysis that can help you understand it.\n'
        img_lst = []
        for i, r in enumerate(ref):
            input_ids = r.metadata['input_ids'][0]
            label = r.metadata['labels'][0]
            # print(input_ids)
            ques = tokenizer.decode(input_ids)
            ans = tokenizer.decode(label)
            image = r.metadata['origin_image_feature'][0]
            ref_str += f"Example {i+1}:\n{ques[:-4]}\nAnwser:{ans[:-4]}\n\n"
            img_lst.append(image)
            # ref_lst.append(r.metadata)
            # print(ref_str)
            # print(image.shape)
            # input('-=-' * 20)
        return ref_str, img_lst
    
class Eval:

    def __init__(self, type='loop', examples=[], data_path=None, num_examples=None, config=None, task=''):
        """
        Args:
            type: 'loop' or 'batch'
            examples: a list of examples
            data_path: a path to a csv file
            num_examples: the number of examples to be evaluated
            config: the configuration of the model
        """
        
        if not examples:
            raise ValueError("Either examples or data_path must be provided")

        if config is None:
            config = {
                "model": 'gpt-4o-mini',
                "temperature": 0.5,
                "max_tokens": 1000,
            }
            print(config)
        
        self.model = config['model']
        task = task
        
        self.config = config
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

        assert type in ['batch', 'loop']

        self.type = type
        self.results = None
        timestamp = datetime.datetime.now().strftime("%m%d_%H")
        self.output_path = f'./save/{task}_{model}_{timestamp}.json'
        self.gptreq = None
    
    def multiple_inference(self, instances):
        delete = []
        # if not self.gptreq:
        #     self.gptreq = LoopRequest()
        # res_list = self.gptreq.batch_req(instances, self.config, save=True, save_dir=self.output_path)
        for i, instance in enumerate(instances):
            if 'data_id' in instance:
                del instance['data_id']
            if 'input_feature' in instance:
                del instance['input_feature']
            instance["input_ids"] = torch.tensor([tokenizer(instance["input_ids"], max_length=512, truncation=True)["input_ids"]])
            # print(instance["input_ids"].shape)
            if instance["input_ids"].shape[-1] == 512:
                delete.append(i)
                continue

            instance["attention_mask"] = (instance['input_ids'] != tokenizer.pad_token_id).long()
            instance = {key: value.to(device) for key, value in instance.items()}
            model.to(device)
            # instance["input_ids"] = instance["input_ids"].to(device)
            # instance["attention_mask"] = instance["attention_mask"].to(device)
            # print(instance["input_ids"].device)  # 应该输出 `cuda:0`
            # print(model.device)
            # for k, i in instance.items():
            #     print(k, i.device)
            # print(instance["input_ids"])
            # print(len(instance["input_ids"][0]))
            # input()
            output = self.model(**instance)
            logits = output['logits']
            
            # label = tokenizer.decode(instance['labels'][0][0])
            #2841,  1465,  7163
            #nega,  posi,  neut
            print(logits.shape)
            input()
            softmax_probs = torch.softmax(logits[0][0], dim=-1)
            top_values, top_indices = torch.topk(softmax_probs, k=1, dim=-1)
            ans = tokenizer.decode(top_indices[0])
            # print('lab, pre',i, ans)
            self.examples[i]["Pred"] = ans
            self.examples[i]["PreAnswer"] = ans
            # print(tokenizer.decode(top_indices))
            # print('lab, pre',i, ans)
            self.examples[i]['logprobs'] = torch.log(top_values).detach().cpu()
            # print(len(self.examples))
            # print(self.examples[i])
            # input()
        for i in delete[::-1]:
            del self.examples[i]
        # assert len(res_list) == len(self.examples)

        # for i, s in enumerate(self.examples):
        #     response = res_list[i]['response']
        #     self.examples[i]["Pred"] = response
        #     self.examples[i]["PredAnswer"] = extract_fn(response)
        #     if "logprobs" in res_list[i]:
        #         self.examples[i]["logprobs"] = res_list[i]["logprobs"]
            # self.examples[i]["PredIndex"] = extract_result_index(response)

    def batch_inference(self, instances, extract_fn):
        res_list = batch_query_openai_chat_model(instances, self.config, save_dir=self.output_path)
        assert len(res_list) == len(self.examples)
        
        for i, s in enumerate(self.examples):
            response = res_list[i]['response']
            self.examples[i]["Pred"] = response
            self.examples[i]["PredAnswer"] = extract_fn(response)
    
    def extract_results(self):
        return self.examples

    def eval(self, format_fn, check_fn=None, extract_fn=None):
        print(f'Formating {len(self.examples)} questions ...')
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            instances = list(tqdm(executor.map(format_fn, self.examples), total=len(self.examples)))
        # for row in tqdm(self.examples):
        #     instances.append(format_fn(row))
        # for instance in instances[:5]:
        #     print(instance)
        # input()
        print(f'Begin Inference ...')
        model.to(device)
        if self.type == 'loop':
            self.multiple_inference(instances)
        else:
            self.batch_inference(instances, extract_fn)
        print(len(self.examples)) 
        cors = []
        for i, s in enumerate(self.examples):
            if 'Pred' not in s:
                continue
            print(list(s.keys()))
            print('pre,label', s['Pred'], s['labels'])
            score = 1.0 if s['Pred'] == s["labels"] else 0.0
            cors.append(score)

        acc = np.mean(cors)
        return acc

    def get_results(self):
        return self.examples




@dataclass
class MyArguments():
    """
    Data Arguments
    """
    is_few: bool = field(
        default=False
    )
    few: str = field(
        default='1'
    )
    data_dir: str = field(
        default="datas",
        metadata={"help": "The input data dir. Should contain the .json files (or other data files) for the task."}
    )

    data_name: str = field(
        default="few_shot"
    )

    task_prompt: str = field(
        default=None,
        metadata={"help": "Prompt for task, e.g. Is the sentence against commonsense: "}
    )

    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    random_p: float = field(
        default=0
    )

    shuffle_ra: bool = field(
        default=False, 
    )

    rand_ra: bool = field(
        default=False, 
    )

    ### image ###
    use_image: bool = field(
        default=False,
        metadata={"help": "If use image as knowledge argumentation"}
    )

    image_grounding_path: str = field(
        default="./datas/commongen/bing_image_for_commongen.json", 
        metadata={"help": "Path to grounding image information (.json file). Contain query_to_image_file information"}
    )

    image_input_path: str = field(
        default="./datas/commongen/blip2_image_feats.lmdb", 
        metadata={"help": "Lmdb path of the image feature (.lmdb dir). Not necessary. For accelerate training"}
    )

    image_num: int = field(
        default=0
    )

    ### text ###
    use_text: bool = field(
        default=False,
        metadata={"help": "If use text as knowledge argumentation"}
    )

    text_grounding_path: str = field(
        default="./datas/commongen/bing_text_for_commongen.json", 
        metadata={"help": "Path to grounding text information (.json file). Contain query_to_text_id information"}
    )

    text_input_path: str = field(
        default="./datas/commongen/blip2_text_feats.lmdb", 
        metadata={"help": "Lmdb path of the grounding text input (.lmdb dir). Not necessary. For accelerate training"}
    )

    text_num: int = field(
        default=0
    )

    ### others ###
    debug_mode: bool = field(
        default=False,
    )

    """
    Model Arguments
    """
    whiteboard: str = field(
        default=None,
        metadata={"help": "Useless!"}
    )

    model_name_or_path: str = field(
        default="src/pre_trained_lm/t5-base", 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    integrator_name_or_path: str = field(
        default="src/pre_trained_lm/bert-base-uncased", 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    resume_checkpoint: str = field(
        default=None, 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    num_query_token: int = field(
        default=32
    )

    prompt_length: int = field(
        default=32
    )

    tokens_learning_rate: float = field(
        default=None
    )

    cold_start_step: int = field(
        default=0
    )

    without_query: bool = field(
        default=False
    )

class MydataCollator(DataCollatorCommongen):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        data_id = [e['data_id'] for e in examples]
        batch['data_id'] = data_id
        return batch

parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()
config = AutoConfig.from_pretrained(
    args.model_name_or_path
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path
)

integrator_tokenizer = AutoTokenizer.from_pretrained(
    args.integrator_name_or_path
)
# 加载 LLaMA 3 本地模型
config.use_decoder_only_backbone_model = False
    
tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG]"]})
model = MoreT5(
    config=config,
    args=args,
    num_query_token=args.num_query_token,
    prompt_length=args.prompt_length,
    integrator_path=args.integrator_name_or_path,
    backbone_model=args.model_name_or_path,
    # resume_checkpoint=args.resume_checkpoint,
    tokenizer=tokenizer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.resize_token_embeddings(len(tokenizer))
if args.resume_checkpoint:
    checkpoint = torch.load(args.resume_checkpoint+"/pytorch_model.bin")
    model.load_state_dict(checkpoint, strict=False)
# model.to(device)  
bats = 64
Dataset = CommongenDataset
labeled_train_dataset = Dataset(args, tokenizer, integrator_tokenizer, "train", unlabeled=False)
unlabeled_train_dataset = Dataset(args, tokenizer, integrator_tokenizer, "train", unlabeled=True)
Collator = MydataCollator
data_collator = Collator(tokenizer, integrator_tokenizer)
label_dataloader = torch.utils.data.DataLoader(labeled_train_dataset, collate_fn=data_collator)
unlabel_dataloader = torch.utils.data.DataLoader(unlabeled_train_dataset, collate_fn=data_collator)

embed = model.backbone_model.encoder

label_data = []
unlabel_data = []
# torch.set_printoptions(threshold=float('inf'))

with torch.no_grad():
    for batch in tqdm(label_dataloader):

        # print(len(batch))
        # for k, i in batch.items():
        #     try:
        #         print(k, i.device)
        #     except:
        #         print(k, type(k))
        # for key, value in batch.items():
        #     if key != 'data_id':
        #         batch[key] = value.to(device)
        # inputs = {key: value.to(device) for key, value in batch.items()}
                
        # # 假设你模型输入是 text + image
        # print('batch', batch)
        input_ids = batch['input_ids']        # 如果是已经tokenized的
        image_features = batch['origin_image_feature']      # 图像已处理成tensor
        # print('input_ids', input_ids.shape)
        img_token_id = tokenizer.convert_tokens_to_ids("[IMG]")
        # print('img_token_id', img_token_id)
        # img_token_index = (input_ids == img_token_id).nonzero(as_tuple=True)[0]
        img_positions = (input_ids == img_token_id).unsqueeze(-1).float()
        
        pooled_image_features = [feat.mean(dim=1).squeeze(0) for feat in image_features]  # List[(768,)]

        # 转成一个 tensor，变成 (batch_size, 768)
        image_feature_tensor = torch.stack(pooled_image_features, dim=0)
        image_feature_broadcasted = image_feature_tensor.unsqueeze(1)
        # 放到模型设备上
        # print('image shape', image_feature_broadcasted.shape)
        inputs_embeds = model.backbone_model.encoder.embed_tokens(input_ids)
        # print('inputs_embeds shape', inputs_embeds.shape)
        outputs = inputs_embeds * (1 - img_positions) + image_feature_broadcasted * img_positions

        batch['input_feature'] = outputs
        # bsz = batch['input_ids'].size(0)
        # # for i in range(bsz):
        # #     for k, v in batch.items():
        # #         try:
        # #             print(k, v.shape)
        # #         except:
        # #             print(k, v)
        # print('pre', batch['input_feature'].shape)
        # batch = [{k: v[i] for k, v in batch.items()} for i in range(bsz)]
        # print('lat', batch['input_feature'].shape)
        # for k, v in batch[0].items():
        #     try:
        #         print(k, v.shape)
        #     except:
        #         print(k, v)
        # input()
        # inputs = {key: value.to(device) for key, value in batch.items()}
        label_data.append(batch)
        # print(label_data)
        # print(len(label_data))

with torch.no_grad():
    for batch in tqdm(unlabel_dataloader):
        flag = False
        for key, value in batch.items():
            if value is None:
                flag = True
                break
            # if key != 'data_id':
            #     batch[key] = value.to(device)
        if flag:
            continue
        # 假设你模型输入是 text + image
        # print('batch', batch)
        # inputs = {key: value.to(device) for key, value in batch.items()} 
        input_ids = batch['input_ids']        # 如果是已经tokenized的
        image_features = batch['origin_image_feature']      # 图像已处理成tensor
        print(batch.keys())
        input()
        # print('input_ids', input_ids.shape)
        img_token_id = tokenizer.convert_tokens_to_ids("[IMG]")
        # print('img_token_id', img_token_id)
        # img_token_index = (input_ids == img_token_id).nonzero(as_tuple=True)[0]
        img_positions = (input_ids == img_token_id).unsqueeze(-1).float()
        
        pooled_image_features = [feat.mean(dim=1).squeeze(0) for feat in image_features]  # List[(768,)]

        # 转成一个 tensor，变成 (batch_size, 768)
        image_feature_tensor = torch.stack(pooled_image_features, dim=0)
        image_feature_broadcasted = image_feature_tensor.unsqueeze(1)
        # 放到模型设备上
        # print('image shape', image_feature_broadcasted.shape)
        inputs_embeds = model.backbone_model.encoder.embed_tokens(input_ids)
        # print('inputs_embeds shape', inputs_embeds.shape)
        outputs = inputs_embeds * (1 - img_positions) + image_feature_broadcasted * img_positions

        batch['input_feature'] = outputs
        # bsz = batch['input_ids'].size(0)
        # batch = [{k: v[i] for k, v in batch.items()} for i in range(bsz)]
        # print(batch)
        # inputs = {key: value.to(device) for key, value in batch.items()}
        unlabel_data.append(batch)

print('labeled_data', len(label_data))
# for k, v in label_data[0].items():
#     print(k)
print('unlabeled_data', len(unlabel_data))
task = args.data_name
# exit()
nr = NearestReference(k=2)
nr.embed_data(label_data, f'tmp/{task}_labeled')
input()



inference_times = 4
inference_list = []
inference_eval = []
# num_examples = len(unlabel_data)
infer_config = {
        'type': 'loop',
        'task': task,
        'config': {
            "model": model,
            "temperature": 1,
            "max_tokens": 1000,
            "logprobs": True
        }
    }

for i in range(inference_times):
    print(f"Start inference {i}")
    inference_data = copy.deepcopy(unlabel_data)
    # for key, value in inference_data.items():
    #     if key != 'data_id':
    #         inference_data[key] = value.to(device)
    eval = Eval(examples=inference_data, **infer_config)
    print('ok12412')
    inference_eval.append(eval)
    print('098765678')
    few_shot_acc = eval.eval(format_fn=format_few_shot)
    print('ijhgyuik')
    res_list = eval.get_results()
    del eval
    inference_list.append(res_list)
    # print(res_list[0].keys())
    entropy_list = [calculate_entropy(infer['logprobs']) for infer in res_list]
    del inference_data

conf_samples = []
unconsis_indexs = []
save_data = copy.deepcopy(unlabel_data)

for idx in range(num_examples):
    pred_list = []
    for i in range(inference_times):
        pred = inference_list[i][idx]['PredAnswer']
        if type(pred) == list:
            pred = str(pred[0])
        pred_list.append(pred)

    entropy = calculate_entropy(inference_list[0][idx]['logprobs'])

    if len(set(pred_list)) > 1:
        unconsis_indexs.append(idx)
        save_data[idx]['consist'] = 0
        save_data[idx]['entropy'] = entropy
    else:
        save_data[idx]['PseudoLabel'] = pred_list[0]
        save_data[idx]['consist'] = 1
        save_data[idx]['entropy'] = entropy

        conf_samples.append(save_data[idx])

print(f'Consistent Rate: {len(conf_samples) / num_examples}')
print(f'Drop Rate: {(len(unlabel_data) - len(unconsis_indexs) - len(conf_samples)) / num_examples}')
print(f'Unconsistent Rate: {len(unconsis_indexs) / num_examples}')
