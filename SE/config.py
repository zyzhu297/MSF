from common import *

MODELS_CONFIG = {
    "llama3.1": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "llama3.1-mmlu-labeled": {
        "name": "mmlu_labeled",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
     "llama3.1-mmlu_pro_labeled": {
        "name": "mmlu_pro_labeled",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "llama3.1-mmlu_pro_pseudo": {
        "name": "mmlu_pro_pseudo",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "llama3.1-arc_labeled": {
        "name": "arc_labeled",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "llama3.1-arc_pseudo": {
        "name": "arc_pseudo",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "Reflection-Llama-3.1-8B": {
        "name": "Solshine/reflection-llama-3.1-8B",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "Hermes-3-Llama-3.1-8B": {
        "name": "NousResearch/Hermes-3-Llama-3.1-8B",
        "url": "http://localhost:6003/v1",
        "method": "loop"
    },
    "adaptllm-med": {
        "name": "AdaptLLM/medicine-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "adaptllm-fin": {
        "name": "AdaptLLM/finance-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "instuctpt-med": {
        "name": "AdaptLLM/instuctpt-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "instuctpt-fin": {
        "name": "AdaptLLM/instuctpt-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "memoryllm": {
        "name": "memoryllm",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-arc_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-usmle_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-pubmedqa_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-fpb_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-convfinqa_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    }
}


TASK_CONFIG = {
    'mmlu': 'multiple_choice',
    'mmlu_pro' :'multiple_choice',
    'arc':'multiple_choice',
    'FPB' :'multiple_choice',
    'PubMedQA':'multiple_choice',
    'USMLE':'multiple_choice',
    'ConvFinQA' :'math'
}

FUNCTION_UTILS = {
    'math': {
        'format_fn': format_value_prompt,
        'check_fn': check_answer,
        'few_shot_prompt': FEW_SHOT_VALUE_SYSTEM,
        'reflection_fn': format_reflection_value,
    },
    'multiple_choice': {
        'format_fn': format_multichoice_question,
        'check_fn': check_answer,
        'few_shot_prompt': FEW_SHOT_SYSTEM,
        'reflection_fn': format_reflection,
    }
}

EVAL_UTILS = {
    'math': {
        'format_fn': format_value_question,
        'check_fn': check_answer_value,
        'extract_fn': extract_value,
    },
    'multiple_choice': {
        'format_fn': format_question,
        'check_fn': check_answer,
        'extract_fn': extract_result,
    }
}
