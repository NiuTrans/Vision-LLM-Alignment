from typing import Iterable
import random
import numpy as np
## the following codes are adopted from https://github.com/haotian-liu/LLaVA
## the following codes are adopted from https://github.com/open-mmlab/Multimodal-GPT 
## the following codes are adopted from https://github.com/Luodian/Otter/

DEFAULT_SYSTEM_TOKEN="### System instuction:"
DEFAULT_PROMPT = f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n"


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_HUMAN_TOKEN = "### Human:"
DEFAULT_HUMAN_QUESTION_PRETOKEN = "### Question:"
DEFAULT_QUESTION_TOKEN = "<question>"
DEFAULT_HUMAN_IMAGE_PRETOKEN = "### Image:"

DEFAULT_ASSISTANT_TOKEN = "### Answer:"
DEFAULT_ANSWER_TOKEN = "<answer>"

DEFAULT_ASSISTANT_END_ROUND_TOKEN="<endofchunk>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# llama3
SYSTEM_MESSEGE_LLAMA3 = "<|start_header_id|>system<|end_header_id|>\n\n"+DEFAULT_PROMPT+"<|eot_id|>"
LLAMA3_HUMAN_QUESTION_PRETOKEN = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_HUMAN_QUESTION_PRETOKEN_END = "<|eot_id|>"
LLAMA3_ASSISTANT_TOKEN = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# llama2
SYSTEM_MESSEGE_LLAMA2 = "[INST] <<SYS>>\n"+DEFAULT_PROMPT+"\n<</SYS>>\n\n"
LLAMA2_HUMAN_QUESTION_PRETOKEN = "[INST] "
LLAMA2_HUMAN_QUESTION_PRETOKEN_END = "</s><s>"
LLAMA2_ASSISTANT_TOKEN = " [/INST] "

# vicuna
VICUNA_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
SYSTEM_MESSEGE_VICUNA=""+VICUNA_PROMPT+""
VICUNA_HUMAN_QUESTION_PRETOKEN = "USER: "
VICUNA_HUMAN_QUESTION_PRETOKEN_END = "</s>"
VICUNA_ASSISTANT_TOKEN = " ASSISTANT: "

# llava
SYSTEM_MESSAGE_LLAVA = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
LLAVA_HUMAN_QUESTION_PRETOKEN = "USER: "
LLAVA_ASSISTANT_TOKEN = " ASSISTANT:"

special_token_list = [DEFAULT_IMAGE_TOKEN] # used for easy image # replacement

DEFAULT_LABEL_PADDING_NUM = -100

def add_special_token(tokenizer, model_path=None):
    tokenizer.add_tokens(special_token_list, special_tokens=True)
    if tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    elif "llama-3" in model_path.lower():
        tokenizer.pad_token_id = 128002

    return tokenizer

TEMPLATE = {
    "description": "Template Modified by DeepSpeed Team for Chat.",
    "prompt_qa_with_image": f'''{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n{DEFAULT_HUMAN_QUESTION_PRETOKEN}\n{DEFAULT_QUESTION_TOKEN}\n\n{DEFAULT_ASSISTANT_TOKEN}\n''',
    "prompt_qa_without_image": f'''{DEFAULT_HUMAN_QUESTION_PRETOKEN}\n{DEFAULT_QUESTION_TOKEN}\n\n{DEFAULT_ASSISTANT_TOKEN}\n''',
}

LLAMA_2_TEMPLATE = {
    "description": "Template for the LLaMA2 models.",
    "prompt_qa_with_image": f'''{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n{LLAMA2_HUMAN_QUESTION_PRETOKEN}{DEFAULT_QUESTION_TOKEN}{LLAMA2_ASSISTANT_TOKEN}''',
    "prompt_qa_without_image": f'''{LLAMA2_HUMAN_QUESTION_PRETOKEN}{DEFAULT_QUESTION_TOKEN}{LLAMA2_ASSISTANT_TOKEN}''',
}

LLAMA_3_TEMPLATE = {
    "description": "Template for the LLaMA3 models.",
    "prompt_qa_with_image": f'''{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n{LLAMA3_HUMAN_QUESTION_PRETOKEN}{DEFAULT_QUESTION_TOKEN}{LLAMA3_HUMAN_QUESTION_PRETOKEN_END}{LLAMA3_ASSISTANT_TOKEN}''',
    "prompt_qa_without_image": f'''{LLAMA3_HUMAN_QUESTION_PRETOKEN}{DEFAULT_QUESTION_TOKEN}{LLAMA3_HUMAN_QUESTION_PRETOKEN_END}{LLAMA3_ASSISTANT_TOKEN}''',
}

VICUNA_TEMPLATE = {
    "description": "Template for the vicuna models.",
    "prompt_qa_with_image": f'''{DEFAULT_HUMAN_IMAGE_PRETOKEN}\n{DEFAULT_IMAGE_TOKEN}\n\n{VICUNA_HUMAN_QUESTION_PRETOKEN}{DEFAULT_QUESTION_TOKEN}{VICUNA_ASSISTANT_TOKEN}''',
    "prompt_qa_without_image": f'''{VICUNA_HUMAN_QUESTION_PRETOKEN}{DEFAULT_QUESTION_TOKEN}{VICUNA_ASSISTANT_TOKEN}''',
}

LLAVA_TEMPLATE = {
    "description": "Template for the LlaVA models.",
    "prompt_qa_with_image": f'''{LLAVA_HUMAN_QUESTION_PRETOKEN}{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_QUESTION_TOKEN}{LLAVA_ASSISTANT_TOKEN}''',
    "prompt_qa_without_image": f'''{LLAVA_HUMAN_QUESTION_PRETOKEN}{DEFAULT_QUESTION_TOKEN}{LLAVA_ASSISTANT_TOKEN}''',
}

class Prompter:
    def __call__(self, question, with_image=True, first_message=False, num_images=-1, options=None, template="default"):
        if options:
            raise NotImplementedError("options not supported yet")
            options = ", ".join(options)
            res = TEMPLATE["prompt_choice"].format(image=DEFAULT_IMAGE_TOKEN, question=question, options=options)
        else:
            if with_image:
                if template == "default":
                    res = TEMPLATE["prompt_qa_with_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template == "llama_2":
                    res = LLAMA_2_TEMPLATE["prompt_qa_with_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template == "llama_3":
                    res = LLAMA_3_TEMPLATE["prompt_qa_with_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template == "vicuna":
                    res = VICUNA_TEMPLATE["prompt_qa_with_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template in ["llava", "llava_next"]:
                    res = LLAVA_TEMPLATE["prompt_qa_with_image"].replace(DEFAULT_QUESTION_TOKEN, question)

            else:
                if template == "default":
                    res = TEMPLATE["prompt_qa_without_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template == "llama_2":
                    res = LLAMA_2_TEMPLATE["prompt_qa_without_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template == "llama_3":
                    res = LLAMA_3_TEMPLATE["prompt_qa_without_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template == "vicuna":
                    res = VICUNA_TEMPLATE["prompt_qa_without_image"].replace(DEFAULT_QUESTION_TOKEN, question)
                elif template in ["llava", "llava_next"]:
                    res = LLAVA_TEMPLATE["prompt_qa_without_image"].replace(DEFAULT_QUESTION_TOKEN, question)
            
            if first_message:
                if template == "defalut":
                    res = DEFAULT_PROMPT + res
                elif template == "llama_3":
                    res = SYSTEM_MESSEGE_LLAMA3 + res
                elif template == "llama_2":
                    res = SYSTEM_MESSEGE_LLAMA2 + res.replace(f"{LLAMA2_HUMAN_QUESTION_PRETOKEN}","",1)
                elif template == "vicuna":
                    res = SYSTEM_MESSEGE_VICUNA + " " + res
                elif template in ["llava", "llava_next"]:
                    res = SYSTEM_MESSAGE_LLAVA + " " + res

        return res

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()

def _flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def flatten(items):
    return list(_flatten(items))


def split_list_with_random_num_items_up_to_a_certain_number(input_list, max_num):
    if len(input_list) <= max_num:
        return [input_list]
    else:
        random_num = random.randint(1, max_num)
        return [input_list[:random_num]] + split_list_with_random_num_items_up_to_a_certain_number(input_list[random_num:], max_num)
            
def random_grouping(input_list, max_num):
    # We have deleted the feature of multiple image, so this function is not need.
    return [[item] for item in input_list]