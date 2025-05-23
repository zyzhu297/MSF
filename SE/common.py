import re
import json

"""
Prompt templates for Inference
"""

QUERY_TEMPLATE_MULTICHOICE = """
Answer the multiple choice question. Your response should be of the following format: 'Answer: LETTER' (without quotes).

Question: 
{question}

Options:
{options}

""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-Z])"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

def format_options(options):
    if type(options) == str:
        options = json.loads(options)
    options = [f"{chr(65+i)}. {option}" for i, option in enumerate(options)]
    options_str = "\n".join(options)
    return options_str

def format_multichoice_question(row):
    question = row['question']
    options_str = format_options(row['options'])
    return QUERY_TEMPLATE_MULTICHOICE.format(question=question, options=options_str)

def format_question(row):
    return [{"role": "user", "content": format_multichoice_question(row)}]

def check_answer(res, gt):
    pred = extract_result(res)
    return pred == gt

def extract_result(res):
    match = re.search(ANSWER_PATTERN_MULTICHOICE, res)
    extracted_answer = match.group(1) if match else res[0].upper()
    return extracted_answer

def extract_result_index(res):
    match = re.search(ANSWER_PATTERN_MULTICHOICE, res)
    extracted_answer = ord(match.group(1)) - ord('A') if match else None
    return extracted_answer

"""
Prompt templates for Value Inference
"""

QUERY_TEMPLATE_VALUE = """
Answer the following math question, output the value should in format: 'Answer: VALUE \n' (without quotes, VALUE should be digits).

Question: 
{question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.

""".strip()

VALUE_PATTERRN = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?[%]*"

def format_value_prompt(row):
    question = row['question']
    return QUERY_TEMPLATE_VALUE.format(question=question)

def format_value_question(row):
    return [{"role": "user", "content": format_value_prompt(row)}]


def normoalize_num(num):
    def eval_num(num):
        num = num.replace('%','/100').replace(',','')
        try:
            num = eval(num)
        except Exception as e:
            num = float('inf')
            pass
        return num
    val_reg = re.compile(VALUE_PATTERRN)
    return [eval_num(num) for num in val_reg.findall(num)]


def extract_value(res):
    ans_reg = re.compile(ANSWER_PATTERN)
    ans_set = ans_reg.findall(res)
    vals = []
    for ans in ans_set:
        vals += normoalize_num(ans)
    return vals

def check_value_equal(res_arr, gt_arr):
    import math
    for gt_num in gt_arr:
        for pred_num in res_arr:
            if math.isclose(pred_num, gt_num, rel_tol=1e-2):
                return True
    return False

def check_answer_value(res, gt):
    pred = extract_value(res)
    gt = normoalize_num(gt)
    return check_value_equal(pred, gt)

"""
Prompt templates for Self-Reflection and Pseudo-Labeling
"""

FEW_SHOT_SYSTEM = """
You are an expert in the multiple choice question. Below are some examples of questions and their corresponding answer.

{reference}
""".strip()

FEW_SHOT_VALUE_SYSTEM = """
You are an expert in the math question. Below are some examples of questions and extracted answer.

{reference}
""".strip()

REFLECTION = """Here are the multiple answers of the multiple choice question.  Please consider them thoroughly and give me the correct answer. Your response should be of the following format: 'Answer: LETTER' (without quotes).

Question: 
{question}

Options:
{options}

Multiple Answers:
{answers}

Now, please give me the final correct answer:
"""

REFLECTION_VALUE = """Here are the multiple answers of the math question.  Please consider them thoroughly and give me the correct answer. Your response should be of the following format: 'Answer: VALUE \n' (without quotes, VALUE should be digits).

Question: 
{question}

Multiple Answers:
{answers}

Now, please give me the final correct answer:
"""

def format_question_and_answer(row):
    question = row['question']
    question_str = f'Question: {question}\n'
    options = format_options(row['options']) if 'options' in row else ''
    options_str = f'Options:\n{options}\n' if options else ''
    answer = row['answer']
    answer_str = f'Answer: {answer}'
    return f"{question_str}{options_str}{answer_str}"

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