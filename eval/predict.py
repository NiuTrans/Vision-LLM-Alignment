#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
from platform import processor
import sys
from tqdm import tqdm
import json


import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from transformers import AutoTokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) + "/training")

from training.ppo_training.ppo_training_utils import sampling, sampling_llava
from training.utils.data import build_dataset, DataCollatorPadToMaxLenForPrediction
from training.utils.utils import print_rank_0, to_device, set_random_seed, get_all_reduce_mean
from training.utils.ds_utils import get_train_ds_config
from training.utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, fuse_lora, unfuse_lora
from training.utils.model import create_dsvl_model_and_transforms, build_model

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a multi-modal task")

    parser.add_argument('--data_path',
                        type=str,
                        default='./data/',
                        help='Where the training data are stored.')
    
    parser.add_argument('--image_folder',
                        type=str,
                        default='./data/',
                        help='Where the image data are stored.')

    parser.add_argument('--data_debug_path',
                        type=str,
                        default=None,
                        help='If provided, will save 10 training samples'
                        'to the path for debug purpose.')

    parser.add_argument('--dataset_names',
                        nargs='*',
                        default=['minigpt4'],
                        help='Name of training dataset(s) to be used. Accepted format:'
                        '1) a single dataset name, 2) multiple dataset names in the'
                        'form: dataset1 dataset2 ...')

    parser.add_argument('--dataset_samples',
                        nargs='*',
                        default=['all'],
                        help='How many samples do we use from each dataset.'
                        'Should be either a integer number or string all which'
                        'means use all samples. For example: all 512 means'
                        'using all samples form first data and 512 samples'
                        'from second data')
    
    parser.add_argument('--dataset_concatenate_samples',
                        nargs='*',
                        default=[1],
                        help='How many samples do we concatenate from each dataset.'
                        'Should be either a integer number or string. 1 which'
                        'means use 1 sample for each datapoint')
    
    parser.add_argument(
        "--max_num_image_per_sample",
        type=int,
        default=1,
        help="The maximum number of images per sample.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="The maximum sequence length, note that image tokens are included.",
    )
    
    parser.add_argument('--enable_tensorboard',
                    action='store_true',
                    help='Enable tensorboard logging')
    parser.add_argument(
        "--enable_mmca_attention",
        action='store_true',
        help="enable the new proposed attn, which is similar to cross attention",
    )
    parser.add_argument("--output_path",
                        type=str,
                        default=None,
                        help="Where to store the result")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="default",
        help=
        "Architecture of pretrained model or model identifier from huggingface.co/models.",
    )   
    parser.add_argument(
        "--lm_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--vision_model_name_or_path", 
        default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument(
        "--vis_proj",
        type=str,
        default='baseline',
        help="[baseline, vit, or perceiver], used to projection vision feature to LLM embedding",
    )
    # deepspeed features
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help=
        "FP16 or BF16 precision. FP16 is recommended for typical use cases. BF16 is good for large models",
    )
    ## LoRA for efficient training setting
    parser.add_argument("--lang_lora_dim",
                        type=int,
                        default=0,
                        help="Use LoRA for fine-tuning language decoder (> 0).")
    parser.add_argument("--lang_lora_module_name",
                        type=str,
                        default="model.layers.",
                        help="The scope name of the target LoRA parameters.")
    parser.add_argument("--vis_lora_dim",
                        type=int,
                        default=0,
                        help="Use LoRA for fine-tuning visual encoder (> 0).")
    parser.add_argument("--vis_lora_module_name",
                        type=str,
                        default="encoder.layers.",
                        help="The scope name of the target LoRA parameters.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')

    ## from ppo training
    parser.add_argument('--from_checkpoint',
                        type=str,
                        required=True,
                        help='Path to the trained SFT model.')
    parser.add_argument(
        "--vision_reward_model_name_or_path", 
        default="openai/clip-vit-large-patch14", 
        help=
        "Path to the used vision model for training reward models.",
        type=str)
    parser.add_argument(
        '--vis_encoder_update',
        action='store_true',
        help='Enable vision encoder update.')
    parser.add_argument(
        '--lang_decoder_update',
        action='store_true',
        help='Enable LLM update.')
    
    parser.add_argument('--template',
                        type=str,
                        choices=["default", "llama_2", "llama_3", "llama_3", "vicuna", "llava"],)
    
    parser.add_argument(
        "--max_new_tokens",
        default=384,
        type=int
    )
    parser.add_argument(
        "--topk",
        default=50,
        type=int
    )
    parser.add_argument(
        "--topp",
        default=0.95,
        type=float
    )
    parser.add_argument(
        "--do_sample",
        action="store_true"
    )
    parser.add_argument(
        "--num_return_sequences",
        default=1,
        type=int
    )
    parser.add_argument(
        "--temperature",
        default=0.75,
        type=float
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if 'qwen' in args.vision_model_name_or_path.lower():
        assert args.vis_proj == 'baseline', "qwen's model only support baseline vis_proj as it has the perceiver module inside"

    assert args.batch_size == 1, "For the time being, we only support cases where batchsize is 1."
    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()
    if args.model_architecture == "default":
        tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path,
                                                fast_tokenizer=True)
        tokenizer.padding_side = 'left'
    else:
        tokenizer = None
    
    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    ds_config = get_train_ds_config(
        offload=False,
        args=args,
        stage=2)

    print_rank_0("load model............")

    model, image_processor, tokenizer = build_model(
                                        text_tokenizer=tokenizer,
                                        args=args,
                                        ds_config=ds_config)
    

    if args.model_architecture == "default":
        model.load_state_dict(torch.load(os.path.join(args.from_checkpoint, 'pytorch_model.bin'), map_location='cpu'), strict=False) 
    model.to('cuda')
    
    # Prepare the data
    if len(args.dataset_samples) < len(args.dataset_names):
        assert len(args.dataset_samples) == 1, "when args.dataset_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_samples =  [args.dataset_samples[0]] * len(args.dataset_names)
    if len(args.dataset_concatenate_samples) < len(args.dataset_names):
        assert len(args.dataset_concatenate_samples) == 1, "when args.dataset_concatenate_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_concatenate_samples =  [args.dataset_concatenate_samples[0]] * len(args.dataset_names)
    # convert to int
    args.dataset_concatenate_samples = [int(i) for i in args.dataset_concatenate_samples]

    dataset = build_dataset(
        args.data_path,
        args.data_debug_path,
        args.dataset_names,
        args.dataset_samples,
        args.dataset_concatenate_samples,
        args.max_num_image_per_sample,
        vis_root=args.image_folder,
        vis_processor=image_processor,
        tokenizer=tokenizer,
        template=args.template
    )

    generation_kwargs={
        "topk": args.topk,
        "topp": args.topp,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences,
        "temperature": args.temperature
    }


    # split the dataset into train and evaluation
    train_dataset = dataset
    args.batch_size = 1

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=False, drop_last=False),
        collate_fn=DataCollatorPadToMaxLenForPrediction(args.max_seq_len, tokenizer.pad_token_id, image_processor.crop_size),
    )

    reference = json.load(open(args.data_path, "r", encoding="utf-8"))

    reference_dict = {}
    for item in reference:
        reference_dict[str(item['id'])] = item

    print_rank_0("***** Running predicting *****", args.global_rank)

    model.eval()

    ref_count = 0
    for idx, batch in enumerate(tqdm(train_dataloader)):
        batch = to_device(batch, device)  #torch.size(1, 3, 224, 224]) #torch.Size([1, 1, 3, 224, 224])
        images = batch["image"].half() 
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        image_num = batch["image_num"]
        
        if image_num[0] == 0:
            images = [None]

        if args.model_architecture == 'default':
            sampling_ans = sampling(model, 
                                    images, input_ids, 
                                    attention_mask=attention_mask, 
                                    pad_token_id=tokenizer.pad_token_id,
                                    **generation_kwargs)
        elif args.model_architecture == 'llava':
            sampling_ans = sampling_llava(model, 
                                    images, input_ids, 
                                    attention_mask=attention_mask, 
                                    pad_token_id=tokenizer.pad_token_id,
                                    processor=tokenizer,
                                    **generation_kwargs)
        else:
            raise NotImplementedError("Not support newly added model architecture")
        
        with open(args.output_path, "a") as trg:
            for i in range(args.batch_size):       
                id = str(batch['id'][0])
                ref_image = reference_dict[id]['image'] if reference_dict[id]['image'] is not None else "None"
                ref_label = reference_dict[id]['label'] if 'label' in reference_dict[id].keys() else "None"
                line = " ||| ".join([id, ref_image.strip(),
                        tokenizer.decode(input_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True).replace("\n", "\\n"), 
                        sampling_ans[i][1].replace("\n", "\\n"), 
                        ref_label.strip()]) + "\n"
                print(line)
                trg.write(line)
                ref_count = ref_count + 1
     

if __name__ == "__main__":
    main()