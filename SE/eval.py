import random
import re
import pandas as pd
import sys
sys.path.append('..')
import json
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from common import *
from utils import *
from config import *
import datetime


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
        
        if data_path:
            df = pd.read_csv(data_path)
            examples = [row.to_dict() for _, row in df.iterrows()]
        elif not examples:
            raise ValueError("Either examples or data_path must be provided")

        if config is None:
            config = {
                "model": 'gpt-4o-mini',
                "temperature": 0.5,
                "max_tokens": 1000,
            }
            print(config)
        
        model = config['model'].replace('/', '_')
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
    
    def multiple_inference(self, instances, extract_fn):
        if not self.gptreq:
            self.gptreq = LoopRequest()
        res_list = self.gptreq.batch_req(instances, self.config, save=True, save_dir=self.output_path)
        print(res_list)
        input()
        assert len(res_list) == len(self.examples)

        for i, s in enumerate(self.examples):
            response = res_list[i]['response']
            self.examples[i]["Pred"] = response
            self.examples[i]["PredAnswer"] = extract_fn(response)
            if "logprobs" in res_list[i]:
                self.examples[i]["logprobs"] = res_list[i]["logprobs"]
            # self.examples[i]["PredIndex"] = extract_result_index(response)

    def batch_inference(self, instances, extract_fn):
        res_list = batch_query_openai_chat_model(instances, self.config, save_dir=self.output_path)
        print(res_list)
        input()
        assert len(res_list) == len(self.examples)
        
        for i, s in enumerate(self.examples):
            response = res_list[i]['response']
            self.examples[i]["Pred"] = response
            self.examples[i]["PredAnswer"] = extract_fn(response)
    
    def extract_results(self):
        return self.examples

    def eval(self, format_fn=format_question, check_fn=check_answer, extract_fn=extract_result):
        print(f'Formating {len(self.examples)} questions ...')
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            instances = list(tqdm(executor.map(format_fn, self.examples), total=len(self.examples)))

        input('instances', instances)
        input()
        # for row in tqdm(self.examples):
        #     instances.append(format_fn(row))

        print(f'Begin Inference ...')

        if self.type == 'loop':
            self.multiple_inference(instances, extract_fn)
        else:
            self.batch_inference(instances, extract_fn)
        
        cors = []
        for i, s in enumerate(self.examples):
            score = 1.0 if check_fn(s['Pred'], s["answer"]) else 0.0
            cors.append(score)

        acc = np.mean(cors)
        return acc

    def get_results(self):
        return self.examples

    
if __name__ == "__main__":
    global_res_file = 'global_res.txt'

    task_list = ['mmlu']
    model_list = ['gpt-4o-mini']

    for task in task_list:
        for model in model_list:

            assert task in TASK_CONFIG and model in MODELS_CONFIG

            model_config = MODELS_CONFIG[model]
            eval_config = EVAL_UTILS[TASK_CONFIG[task]]
            os.environ['LLM_BASE_URL'] = model_config["url"]
            infer_config = {
                "model": model_config["name"],
                "temperature": 0.5,
                "max_tokens": 1000,
                # "logprobs": True ## Output logprobs to calculate the entropy of the response.
            }

            eval = Eval(type=model_config["method"], data_path=f'./data/{task}/test.csv', config=infer_config, task=task)
            print(f'Evaluate {task} with {model} ...')

            acc = eval.eval(**eval_config)

            print(f'Accuracy: {acc}') 