import argparse
import torch
import json
from transformers import AutoTokenizer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image

import requests
from PIL import Image
from io import BytesIO

from training.utils.model import create_dsvl_model_and_transforms, build_model
from training.ppo_training.ppo_training_utils import sampling, sampling_llava

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = "llava_v1.5"

    generation_kwargs={
        "topk": 50,
        "topp": 0.95,
        "do_sample": True,
        "max_new_tokens": 1024,
        "temperature": 0.2
    }

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                fast_tokenizer=True)
    args_tmp = {
        'from_checkpoint': args.model_path,
        'model_architecture': 'llava',
        'lang_decoder_update': False
    }
    args_tmp = argparse.Namespace(**args_tmp)

    model, image_processor, tokenizer = build_model(
                                        text_tokenizer=tokenizer,
                                        args=args_tmp,
                                        ds_config=None)

    model.to('cuda')

    print(tokenizer)

    test_path = args.test_path
    test = json.load(open(args.test_path, "r", encoding="utf-8"))
    
    output_path = args.output_path
    output_file = open(output_path, "w", encoding='utf-8')
    
    for item in test:
        qs = item["conversations"][0]["value"]

        image_file = args.image_file+"/"+item["image"]

        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = load_image(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        attention_mask = torch.ones_like(input_ids)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            # output_ids = model.generate(
            #     input_ids,
            #     images=image_tensor,
            #     do_sample=True,
            #     temperature=0.2,
            #     max_new_tokens=1024,
            #     use_cache=True,
            #     stopping_criteria=[stopping_criteria])
            input_ids[input_ids==-200]=32000
            sampling_ans = sampling_llava(model, 
                            image_tensor, input_ids, 
                            attention_mask=attention_mask, 
                            pad_token_id=tokenizer.pad_token_id,
                            processor=tokenizer,
                            **generation_kwargs)

        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # outputs = outputs.strip()
        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        # outputs = outputs.strip()
        output = sampling_ans[0][1]
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).replace("\n", "\\n"))
        print("##################")
        print(output)
        output_file.write(" ||| ".join([item["id"], item["image"],
                    tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).replace("\n", "\\n"), 
                    output.replace("\n", "\\n"), 
                    item["conversations"][1]["value"].strip().replace("\n", "\\n")]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()
    eval_model(args)