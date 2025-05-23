from transformers import HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, AutoTokenizer, T5ForConditionalGeneration
from datasets import load_metric 
import datasets
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import os
import sys
sys.path.append('/home/zyzhu/MORE')
import pandas as pd
import torch
from dataclasses import dataclass, field
import random
import numpy as np
import json

from model.more_t5 import MoreT5
from data.commongen import CommongenDataset, DataCollatorCommongen
from metric.commongen_metric import commongen_metric_builder
from metric.metric import few_metric_builder

# from torch.utils.data import DataLoader
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

import matplotlib.pyplot as plt
from transformers import TrainerCallback

class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            pass
        if "loss" in logs:
            self.losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

loss_plot_callback = LossPlotCallback()

class MySeq2SeqTrainer(Seq2SeqTrainer):
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

@dataclass
class MyArguments():
    """
    Data Arguments
    """
    is_few: bool = field(
        default=True
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


def main():
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
    Dataset = CommongenDataset
    train_dataset = Dataset(args, tokenizer, integrator_tokenizer, "train")
    eval_dataset = Dataset(args, tokenizer, integrator_tokenizer, "val")
    print('length of train_set', len(train_dataset))
    print('length of eval_set', len(eval_dataset))
    # check which sample is unavailable
    # cur_path = os.path.join(args.data_dir, '_'.join([args.data_name, 'train', 'few1']) + '.json')
    # with open(cur_path, 'r') as f:
    #     cur_data = json.load(f)
    # print(cur_data)
    # for i in range(len(eval_dataset)):
    #     t = eval_dataset.__getitem__(i)['data_id']
    #     t = '_'.join(t.split('_')[:-1])
    #     if t in cur_data:
    #         del cur_data[t]
    # print(cur_data)
    # input()
    # print(eval_dataset.__getitem__(0))
    # input()

    Collator = DataCollatorCommongen
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
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_fn if training_args.predict_with_generate else None,
        callbacks=[loss_plot_callback],
    )
    if training_args.do_train:
        logger.info("*** Train ***")
        trainer.train(training_args.resume_from_checkpoint)
        trainer.save_model()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_results = trainer.predict(
            eval_dataset, metric_key_prefix="eval",
        )
        # print(eval_results)
        # input()
        metrics = eval_results.metrics
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                # print(eval_results.predictions)
                predictions = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                
                output_prediction_file = os.path.join(training_args.output_dir, "eval_generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
    # print(training_args)
    # print(training_args.do_pred) #output is no attribute
    if training_args.do_predict:
        # metric_fn = commongen_metric_builder(tokenizer, os.path.join(args.data_dir, "commongen.test.src_alpha.txt"), os.path.join(args.data_dir, "commongen.test.tgt.txt"))
        metric_fn = few_metric_builder(args, tokenizer, 'test')
        trainer = MySeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=metric_fn if training_args.predict_with_generate else None,
        )

        test_dataset = Dataset(args, tokenizer, integrator_tokenizer, "test")
        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="test",
        )
        # print(logits)
        # print('pred'*21)
        # print(predict_results)
        # exit()

        metrics = predict_results.metrics
        metrics["predict_samples"] = len(test_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "test_generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
    plt.plot(loss_plot_callback.losses, label='train')
    plt.plot(loss_plot_callback.eval_losses, label='dev')
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.savefig("{}.png".format(args.output_dir.split('/')[-1]), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
