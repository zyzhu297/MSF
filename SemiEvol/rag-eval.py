import random
import pandas as pd
import os
import sys
sys.path.append('..')
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
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

GLOBAL_RETRIEVAL_CACHE = ThreadSafeDict()

def format_few_shot(data):
    ref_str = nr.fewshot(data)
    system_prompt = FUNCTION_UTILS[question_type]['few_shot_prompt'].format(reference=ref_str)
    user_prompt = FUNCTION_UTILS[question_type]['format_fn'](data)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def format_reflection(data):
    preds = data['Preds']
    ans = data['PredAnswers']
    ans_ref_str = ''
    for i in range(len(ans)):
        ans_ref_str += f"Answer {i+1}: {ans[i]}\nReason: {preds[i]}\n\n"
    user_prompt = REFLECTION.format(question=data['question'], options=format_options(data['options']), answers=ans_ref_str)
    return [{"role": "user", "content": user_prompt}]

def format_reflection_value(data):
    preds = data['Preds']
    ans = data['PredAnswers']
    ans_ref_str = ''
    for i in range(len(ans)):
        ans_ref_str += f"Answer {i+1}: {ans[i]}\nReason: {preds[i]}\n\n"
    user_prompt = REFLECTION_VALUE.format(question=data['question'], answers=ans_ref_str)
    return [{"role": "user", "content": user_prompt}]

class NearestReference:
    def __init__(self, k=4, embed_type='faiss') -> None:
        self.vectorstore = None
        self.selector = None
        self.k = k
        self.embed_type = embed_type

    def read_data(self, data_path):
        df = pd.read_csv(data_path)
        examples = [row.to_dict() for _, row in df.iterrows()]
        return examples

    def embed_data_path(self, data_path):
        data = self.read_data(data_path)
        return self.embed_data(data)

    def embed_data(self, data):
        data_str = [row['question'] for row in data]
        print(f'embed_type: {self.embed_type}')
        if self.embed_type == 'bm25':
            self.retriever = BM25Retriever.from_texts(data_str, metadatas=data)
            self.retriever.k = self.k
        else:
            self.vectorstore = FAISS.from_texts(data_str, embed, metadatas=data)
            self.retriever = SemanticSimilarityExampleSelector(
                vectorstore=self.vectorstore, k=self.k
            )
        return self.retriever

    @retry
    @func_set_timeout(5)
    def retrieve(self, question):
        if self.embed_type == 'bm25':
            res = self.retriever.get_relevant_documents(question)
            res = [r.metadata for r in res]
        else:
            res = self.retriever.select_examples({'question': question})
        return res

    def fewshot(self, question):
        ref = self.retrieve(question['question'])
        ref_str = ''
        for i, r in enumerate(ref):
            ref_str += f"Example {i+1}:\n{format_question_and_answer(r)}\n\n"
        return ref_str

if __name__ == '__main__':
    task_list = ['ConvFinQA']

    for task in task_list:
        model='gpt-4o-mini'

        assert task in TASK_CONFIG and model in MODELS_CONFIG
        embed_type = 'faiss'
        question_type = TASK_CONFIG[task]
        label_path = f'data/{task}/labeled.csv'
        infer_path = f'data/{task}/test.csv'
        eval_config = EVAL_UTILS[question_type]
        eval_config['format_fn'] = format_few_shot
        model_config = MODELS_CONFIG[model]
        os.environ['LLM_BASE_URL'] = model_config["url"]
        infer_config = {
            'type': model_config["method"],
            'task': task,
            'config': {
                "model": model_config['name'],
                "temperature": 0.5,
                "max_tokens": 1000,
            }
        }

        save_path = f'data/{task}/pseudo_{model}.csv'
        embed = OpenAIEmbeddings(model="text-embedding-3-small")

        nr = NearestReference(k=1, embed_type=embed_type)
        nr.embed_data_path(label_path)
        
        infer_df = pd.read_csv(infer_path)
        infer_data = [row.to_dict() for _, row in infer_df.iterrows()]
       
        eval = Eval(examples=infer_data, **infer_config)
        eval_acc = eval.eval(**eval_config)

        print(f'Inference Accuracy: {eval_acc}')