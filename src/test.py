from transformers import HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, AutoTokenizer, T5ForConditionalGeneration
from datasets import load_metric 
import datasets
import requests
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import logging
import os
import sys
import json
sys.path.append('/home/zyzhu/MORE')
import pandas as pd
import torch
from dataclasses import dataclass, field
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model.more_t5 import MoreT5
from data.commongen import CommongenDataset, DataCollatorCommongen
from metric.commongen_metric import commongen_metric_builder
from metric.metric import few_metric_builder

from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

os.environ["WANDB_DISABLED"] = "true"

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import ShardedDDPOption
from transformers.integrations import is_fairscale_available
from transformers.dependency_versions_check import dep_version_check
if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.optim import OSS

from transformers.file_utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
from transformers.trainer_pt_utils import get_parameter_names


class MydataCollator(DataCollatorCommongen):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        data_id = [e['data_id'] for e in examples]
        batch['data_id'] = data_id[0]
        return batch
        

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


url = "http://localhost:8000/v1/chat/completions"
parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()
setattr(training_args, "tokens_learning_rate", args.tokens_learning_rate)
setattr(training_args, "cold_start_step", args.cold_start_step)

if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )
args.output_dir = training_args.output_dir 
print(args)
print(training_args)
# exit()

config = AutoConfig.from_pretrained(
    args.model_name_or_path
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path
)

integrator_tokenizer = AutoTokenizer.from_pretrained(
    args.integrator_name_or_path
)

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
model.resize_token_embeddings(len(tokenizer))
if args.resume_checkpoint:
    checkpoint = torch.load(args.resume_checkpoint+"/pytorch_model.bin")
    model.load_state_dict(checkpoint, strict=False)
# model.resize_token_embeddings(len(tokenizer))
Dataset = CommongenDataset
train_dataset = Dataset(args, tokenizer, integrator_tokenizer, "train")
Collator = MydataCollator
data_collator = Collator(tokenizer, integrator_tokenizer)
def tensor_to_json(tensor):
    return tensor.detach().cpu().tolist()
for split in ['train']:
    dataloder = torch.utils.data.DataLoader(locals()[f"{split}_dataset"], collate_fn=data_collator)
    print('length of data', len(locals()[f"{split}_dataset"]))
    cur_few_path = os.path.join(args.data_dir, '_'.join([args.data_name, split, 'few' + args.few + '.json']))
    cur_path = os.path.join(args.data_dir, '_'.join([args.data_name, split]) + '.json')
    nxt_few_path = os.path.join(args.data_dir, '_'.join([args.data_name, split, 'few' + str(int(args.few) + 1) + '.json']))
    with open(cur_few_path, 'r') as f:
        few_data = json.load(f)
    with open(cur_path, 'r') as f:
        entire_data = json.load(f)
    selected_sample = set(few_data.keys())
    print('=' * 10 + 'selected sample' + '=' * 10)
    print(selected_sample)
    length = len(selected_sample) // 3 + 1
    if split == 'dev':
        length //= 8
    # exit()
    for datas in tqdm(dataloder):
        del datas['data_id']
        # for k, v in datas.items():
            # print(k, type(v), v)
        data = {key: tensor_to_json(value) for key, value in datas.items()}
        
        response = requests.post(url, json=data)
        # print(response)
        # print('-=-=' * 20)
        if response.status_code == 200:
            result = response.json()
            print("✅ API 调用成功，返回结果：")
            # print(result["choices"][0]["message"]["content"])
            print(result)
        else:
            print("❌ API 调用失败，状态码：", response.status_code)
            print(response.text)