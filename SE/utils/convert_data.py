import pandas as pd
import json
from common import *

def format_question_alpaca(row, format_fn=format_multichoice_question):
    input_text = format_fn(row)
    output_test = f'Answer: {row["answer"]}'
    return {
        "instruction": input_text,
        "input": '',
        "output": output_test
    }

def format_qa_gpt(row, format_fn=format_multichoice_question):
    return {
        'messages': [
            {"role": "user", "content": format_fn(row)},
            {"role": "assistant", "content": "Answer: " + row["answer"]}
        ]
    }

def format_gpt_eval(row, format_fn=format_multichoice_question):
    return {
        'question': format_fn(row),
        'answer': f'Answer: {row["answer"]}'
    }

if __name__ == '__main__':
    task = 'mmlu_pro'
    type = 'alpaca'
    datatype = 'labeled'

    input_file = f'./data/{task}/{datatype}.csv'
    output_file = f'./data/{task}/{task}_{datatype}_{type}.jsonl' if 'gpt' in type else f'./data/{task}/{datatype}_{type}.json'

    df = pd.read_csv(input_file)
    if type == 'alpaca':
        examples = [format_question_alpaca(row, format_multichoice_question) for _, row in df.iterrows()]
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
    elif type == 'gpt':
        examples = [format_qa_gpt(row, format_multichoice_question) for _, row in df.iterrows()]
        with open(output_file, 'w') as f:
            for obj in examples:
                f.write(json.dumps(obj) + '\n')
    elif type == 'gpt_eval':
        examples = [format_gpt_eval(row, format_multichoice_question) for _, row in df.iterrows()]
        with open(output_file, 'w') as f:
            for obj in examples:
                f.write(json.dumps(obj) + '\n')
    
    print(f'Finished formatting {len(examples)} examples to {output_file}')