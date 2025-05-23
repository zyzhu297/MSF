import random
import pandas as pd
import os
import sys
sys.path.append('..')
sys.path.append('/home/zyzhu/SemiEvol')
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
import copy
import numpy as np
from common import *
from eval import Eval
import threading
from retrying import retry
from func_timeout import func_set_timeout
from config import *

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
    ref_str = nr.fewshot(data)
    system_prompt = FUNCTION_UTILS[question_type]['few_shot_prompt'].format(reference=ref_str)
    user_prompt = FUNCTION_UTILS[question_type]['format_fn'](data)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

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
        if os.path.exists(embed_path):
            self.vectorstore = FAISS.load_local(embed_path, embed)
            # , allow_dangerous_deserialization=True)
        else:
            os.makedirs(embed_path, exist_ok=True)
            data_str = [row['question'] for row in data]
            self.vectorstore = FAISS.from_texts(data_str, embed, metadatas=data)
            self.vectorstore.save_local(embed_path)

        self.selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore, k=self.k
        )
        return self.vectorstore

    @retry
    @func_set_timeout(2)
    def retrieve(self, question):
        cached_result = GLOBAL_RETRIEVAL_CACHE.get(question)
        if cached_result is not None:
            return cached_result
        res = self.selector.select_examples({'question': question})
        GLOBAL_RETRIEVAL_CACHE.set(question, res)
        return res

    def fewshot(self, question):
        ref = self.retrieve(question['question'])
        ref_str = ''
        for i, r in enumerate(ref):
            ref_str += f"Example {i+1}:\n{format_question_and_answer(r)}\n\n"
        print('ref_str', ref_str)
        return ref_str

if __name__ == '__main__':
    
    task = 'FPB'
    model='gpt-4o-mini'

    assert task in TASK_CONFIG and model in MODELS_CONFIG

    question_type = TASK_CONFIG[task]
    label_path = f'data/{task}/labeled.csv'
    unlabel_path = f'data/{task}/unlabeled.csv'
    model_config = MODELS_CONFIG[model]
    os.environ['LLM_BASE_URL'] = model_config["url"]

    infer_config = {
        'type': model_config["method"],
        'task': task,
        'config': {
            "model": model_config['name'],
            "temperature": 1,
            "max_tokens": 1000,
            "logprobs": True
        }
    }
    eval_config = EVAL_UTILS[question_type]
    eval_config['format_fn'] = format_few_shot

    save_path = f'data/{task}/pseudo_{model}.csv'
    embed = OpenAIEmbeddings(model="text-embedding-3-small")

    nr = NearestReference(k=3)
    nr.embed_data_path(label_path, f'tmp/{task}_labeled')
    
    unlabel_df = pd.read_csv(unlabel_path)
    unlabel_data = [row.to_dict() for _, row in unlabel_df.iterrows()]

    # num_examples = len(unlabel_data)
    # unlabel_data = random.Random(0).sample(unlabel_data, num_examples)
    
    save_data = copy.deepcopy(unlabel_data)

    inference_times = 4
    inference_list = []
    inference_eval = []

    for i in range(inference_times):
        print(f"Start inference {i}")
        inference_data = copy.deepcopy(unlabel_data)
        eval = Eval(examples=inference_data, **infer_config)
        inference_eval.append(eval)
        few_shot_acc = eval.eval(**eval_config)
        res_list = eval.get_results()
        inference_list.append(res_list)
        entropy_list = [calculate_entropy(infer['logprobs']) for infer in res_list]

    conf_samples = []
    unconsis_indexs = []

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

    unconsis_examples = []
    for idx in unconsis_indexs:
        unlabel_data[idx]['PredAnswers'] = [inference_list[i][idx]['PredAnswer'] for i in range(inference_times)]
        unlabel_data[idx]['Preds'] = [inference_list[i][idx]['Pred'] for i in range(inference_times)]
        unconsis_examples.append(unlabel_data[idx])

    if unconsis_examples:
        unconsist_eval = Eval(examples=unconsis_examples, **infer_config)
        unconsist_acc = unconsist_eval.eval(**eval_config)
        unconsist_res = unconsist_eval.get_results()
        unconf_unconsit_indexs = []
        
        for idx, res in zip(unconsis_indexs, unconsist_res):
            entropy = calculate_entropy(res['logprobs'])
            save_data[idx]['entropy'] = entropy
            save_data[idx]['PseudoLabel'] = res['PredAnswer']
            conf_samples.append(save_data[idx])

    for i, s in enumerate(save_data):
        score = 1.0 if s["PseudoLabel"] == s["answer"] else 0.0
        save_data[i]["Accuracy"] = score

    save_df = pd.DataFrame(save_data)
    save_df.to_csv(save_path, index=False)
    print(f"Save the final pseudo-labels to {save_path}")
