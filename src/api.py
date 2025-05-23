from transformers import HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, AutoTokenizer, T5ForConditionalGeneration
from datasets import load_metric 
import datasets
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

class MySeq2SeqTrainer(Seq2SeqTrainer):
    # def predict(self, test_dataset, metric_key_prefix="test"):
    #     # 调用父类的 predict 方法，获取结果
    #     outputs = super().predict(test_dataset, metric_key_prefix=metric_key_prefix)
    #     model_outputs = outputs.predictions  # 这取决于Trainer的实现
    #     loss = model_outputs['loss']
    #     logits = model_outputs['logits']
    #     return logits, outputs

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.tokens_learning_rate is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                print("=========")
                for n, p in opt_model.named_parameters():
                    if (n in decay_parameters and p.requires_grad and "prompt_tokens" in n):
                        print(n)
                print("=========")
                
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "prompt_tokens" in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.tokens_learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "prompt_tokens" in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.tokens_learning_rate,
                    },
                    
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "prompt_tokens" not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "prompt_tokens" not in n)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        # for param_group in self.optimizer.param_groups:# 
        #     print('学习率: ',param_group['lr'])
        # exit()

        return self.optimizer

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        if self.args.cold_start_step > 0:
            if self.state.global_step < self.args.cold_start_step:
                p = float(self.state.global_step) / self.args.cold_start_step
                p = 0.5 * (np.sin(np.pi*(p-0.5))+1)
                inputs["cold_start_p"] = 1 - p

        return inputs

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

model = MoreT5(
    config=config,
    args=args,
    num_query_token=args.num_query_token,
    prompt_length=args.prompt_length,
    integrator_path=args.integrator_name_or_path,
    backbone_model=args.model_name_or_path,
    resume_checkpoint=args.resume_checkpoint,
)
# model.resize_token_embeddings(len(tokenizer))
Dataset = CommongenDataset
train_dataset = Dataset(args, tokenizer, integrator_tokenizer, "train")
dev_dataset = Dataset(args, tokenizer, integrator_tokenizer, "dev")
test_dataset = Dataset(args, tokenizer, integrator_tokenizer, "test")
# print(eval_dataset.__getitem__(0))
# input()

Collator = MydataCollator
data_collator = Collator(tokenizer, integrator_tokenizer)
#TODO
# metric_fn = commongen_metric_builder(tokenizer, os.path.join(args.data_dir, "commongen.dev.src_alpha.txt"), os.path.join(args.data_dir, "commongen.dev.tgt.txt"))
metric_fn = few_metric_builder(args, tokenizer, 'dev')
# print(train_dataset[0])
# exit()
trainer = MySeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metric_fn if training_args.predict_with_generate else None,
)
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
    import heapq
    positive_heap = []
    neutral_heap = []
    negative_heap = []
    lst = []
    T = 0.07
    for datas in tqdm(dataloder):
        data = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value for key, value in datas.items()}
        data_id = '_'.join(data["data_id"].split('_')[:-1])
        del data['data_id']
        if data_id in selected_sample:
            continue
        output = model(**data)
        logits = output['logits']
        label = tokenizer.decode(data['labels'][0][0])
        #2841,  1465,  7163
        #nega,  posi,  neut
        softmax_probs = torch.softmax(logits[0][0], dim=-1)
        top_values, top_indices = torch.topk(softmax_probs, k=15, dim=-1)
        ans = tokenizer.decode(top_indices[0])
        # for i in top_indices:
        #     ans.append(tokenizer.decode(i))
        score = function1(top_values, T)
        lst.append((score, (ans, label)))
        # print(score)
        # print([h[0] for h in min_heap])m

from fastapi import FastAPI


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}