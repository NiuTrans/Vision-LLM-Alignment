#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import math
import sys
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
    AutoTokenizer
)

import deepspeed
from transformers import AdamW
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data import build_dataset, DataCollatorPadToMaxLen, split_dataset, shuffle_dataset
from utils.utils import (
    print_rank_0, to_device, 
    save_hf_format, set_random_seed, 
    get_all_reduce_mean, 
    get_optimizer_grouped_parameters, 
    save_zero_three_model
)
from utils.ds_utils import get_train_ds_config
from utils.model import build_model

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Finetune a transformers model on a multi-modal task')

    parser.add_argument('--data_path',
                        type=str,
                        default='./data/',
                        help='Where the training data are stored.')

    parser.add_argument('--data_debug_path',
                        type=str,
                        default=None,
                        help='If provided, will save 10 training samples'
                        'to the path for debug purpose.')
    
    parser.add_argument('--image_folder',
                type=str,
                default=None,
                help='Where the image data are stored.')

    parser.add_argument(
        '--data_train_split_ratio',
        type=float,
        default=0.9,
        help='Ratio of dataset to be splitted as train data. The remaining becomes eval data.',
    )
    parser.add_argument(
        '--is_sft_stage',
        action='store_true',
        help=''
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    
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
        '--max_num_image_per_sample',
        type=int,
        default=8,
        help='The maximum number of images per sample.',
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=2,
        help='Batch size (per device) for the training dataloader.',
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=2,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=4096,
        help='The maximum sequence length, note that image tokens are included.',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help=
        'Initial learning rate (after the potential warmup period) to use.',
    )
    parser.add_argument(
        '--learning_rate_pretraining_components',
        type=float,
        default=0,
        help=
        'Initial learning rate for pre-trained weight, e.g., embedding (after the potential warmup period) to use.',
    )
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.,
                        help='Weight decay to use.')
    parser.add_argument('--num_train_epochs',
                        type=int,
                        default=6,
                        help='Total number of training epochs to perform.')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        'Number of updates steps to accumulate before performing a backward/update pass.',
    )
    parser.add_argument(
        '--lr_scheduler_type',
        type=SchedulerType,
        default='cosine',
        help='The scheduler type to use.',
        choices=[
            'linear', 'cosine', 'cosine_with_restarts', 'polynomial',
            'constant', 'constant_with_warmup'
        ],
    )
    parser.add_argument(
        '--num_warmup_steps',
        type=float,
        default=0,
        help='Number of steps (>1) or ratios (<=1) for the warmup in the lr scheduler.')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Where to store the model.')
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='A seed for reproducible training.')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local_rank for distributed training on gpus')
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        '--lm_model_name_or_path',
        type=str,
        help=
        'Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--vision_model_name_or_path', default='openai/clip-vit-large-patch14', type=str)
    parser.add_argument('--model_architecture', default='default', type=str)
    parser.add_argument(
        '--enable_mmca_attention',
        action='store_true',
        help='enable the new proposed attn, which is similar to cross attention',
    )
    parser.add_argument(
        '--vis_proj',
        type=str,
        default='baseline',
        help='[baseline, vit, or perceiver], used to projection vision feature to LLM embedding',
    )
    # deepspeed features
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp16', 'bf16'],
        default='fp16',
        help=
        'FP16 or BF16 precision. FP16 is recommended for typical use cases. BF16 is good for large models',
    )
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    ## LoRA for efficient training setting
    parser.add_argument('--lang_lora_dim',
                        type=int,
                        default=0,
                        help='Use LoRA for fine-tuning language decoder (> 0).')
    parser.add_argument('--lang_lora_module_name',
                        type=str,
                        default='model.layers.',
                        help='The scope name of the target LoRA parameters.')
    parser.add_argument('--vis_lora_dim',
                        type=int,
                        default=0,
                        help='Use LoRA for fine-tuning visual encoder (> 0).')
    parser.add_argument('--vis_lora_module_name',
                        type=str,
                        default='encoder.layers.',
                        help='The scope name of the target LoRA parameters.')
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument('--template',
                        type=str,
                        choices=['default', 'llama_2', 'llama_3', 'llama_3', 'vicuna', 
                        'llama-3.2-vision'],)
    parser.add_argument(
        '--eval_step',
        type=int,
        default=100,
        help='The evaluation will be conducted every specific number of training steps.')
    parser.add_argument(
        '--from_checkpoint',
        type=str,
        default='./basemodel/',
        help='Specifying the checkpoint directory to be loaded.')
    parser.add_argument(
        '--vis_encoder_update',
        action='store_true',
        help='Enable vision encoder update.')
    parser.add_argument(
        '--lang_decoder_update',
        action='store_true',
        help='Enable LLM update.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.learning_rate_pretraining_components == 0.0:
        # if we do not provide special learning rate, mainly for embedding, the same lr is applied
        args.learning_rate_pretraining_components = args.learning_rate
    assert args.num_warmup_steps >= 0, '--num_warmup_steps must be >= 0'
    if 'qwen' in args.vision_model_name_or_path.lower():
        assert args.vis_proj == 'baseline', "qwen's model only support baseline vis_proj as it has the perceiver module inside"
    return args

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device('cuda')
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(args, offload=False,
                                    stage=2)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()
    if args.model_architecture == 'default':
        tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path,
                                                fast_tokenizer=True)
        tokenizer.padding_side = 'left'
    else:
        tokenizer = None

    model, image_processor, tokenizer = build_model(
            text_tokenizer=tokenizer,
            args=args,
            ds_config=ds_config)
    
    # let load checkpoint 
    if os.path.exists(args.from_checkpoint) and args.model_architecture=='default':
        # we have the deepspeed chekpoint so it is a resumed job
        print_rank_0(f'load checkpoint from {args.from_checkpoint}')
        model.load_state_dict(torch.load(os.path.join(args.from_checkpoint, 'pytorch_model.bin'), map_location='cpu'), strict=False)

    print_rank_0(model, args.global_rank) 
        
    # Prepare the data
    if len(args.dataset_samples) < len(args.dataset_names):
        assert len(args.dataset_samples) == 1, 'when args.dataset_samples is not the same length as args.dataset_names, it should be only one number'
        args.dataset_samples =  [args.dataset_samples[0]] * len(args.dataset_names)
    if len(args.dataset_concatenate_samples) < len(args.dataset_names):
        assert len(args.dataset_concatenate_samples) == 1, 'when args.dataset_concatenate_samples is not the same length as args.dataset_names, it should be only one number'
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
        vis_processor=image_processor,
        vis_root=args.image_folder,
        tokenizer=tokenizer,
        template=args.template
    )
    # split the dataset into train and evaluation
    total_data = len(dataset)
    np_rng = np.random.RandomState(seed=args.seed)
    dataset = shuffle_dataset(dataset, np_rng)
    train_dataset, eval_dataset = split_dataset(dataset, args.data_train_split_ratio)

    if args.model_architecture == "llama-3.2-vision":
        image_size = image_processor.size
    else:
        image_size = image_processor.crop_size

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
        collate_fn=DataCollatorPadToMaxLen(args.max_seq_len, tokenizer.pad_token_id, image_size),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=DistributedSampler(eval_dataset, shuffle=False),
        collate_fn=DataCollatorPadToMaxLen(args.max_seq_len, tokenizer.pad_token_id, image_size),
    )

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, small_lr=args.learning_rate_pretraining_components)

    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.num_warmup_steps <= 1:
        args.num_warmup_steps = int(args.num_warmup_steps * args.num_train_epochs * num_update_steps_per_epoch)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    ds_config = get_train_ds_config(args, offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    start_epoch = 0

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def evaluation(model, eval_dataloader):
        model.eval()
        acc_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                batch = to_device(batch, device)
                images = batch["image"].half() 
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                if args.model_architecture=="default":
                    loss = model(
                        images,
                        input_ids,
                        attention_mask=attention_mask,
                        input_labels=labels,
                        image_num=batch["image_num"],
                    )[0]
                elif args.model_architecture=="llava":
                    loss = model(
                        input_ids=input_ids,
                        pixel_values=images,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=False
                    )[0]
                elif args.model_architecture=="llama-3.2-vision":
                    aspect_ratio_ids = batch["aspect_ratio_ids"]
                    aspect_ratio_mask = batch["aspect_ratio_mask"]
                    images = images.reshape(len(input_ids), -1, images.size(-4), images.size(-3), images.size(-2), images.size(-1))
                    
                    loss = model(
                        input_ids=input_ids,
                        pixel_values=images,
                        aspect_ratio_mask=aspect_ratio_mask,
                        aspect_ratio_ids=aspect_ratio_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=False)[0] 

            acc_loss += loss
        model.train()
        acc_loss = get_all_reduce_mean(acc_loss).item()
        ave_loss = acc_loss / (step + 1)
        print_rank_0(f"the eval average_loss: {ave_loss}", args.global_rank)
        return ave_loss
    
    # Train!
    print_rank_0('***** Running training *****', args.global_rank)
    global_step = 0
    for epoch in range(start_epoch, args.num_train_epochs):
        print_rank_0(
            f'Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}',
            args.global_rank)
        model.train()
        acc_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = to_device(batch, device) 
            images = batch["image"].half() 
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            if args.model_architecture=="default":
                # from utils import pdb;pdb.set_trace()
                loss = model(
                    images,
                    input_ids,
                    attention_mask=attention_mask,
                    input_labels=labels,
                    image_num=batch["image_num"],
                    is_sft_stage=args.is_sft_stage
                )[0]
            elif args.model_architecture=="llava":
                loss = model(
                    input_ids=input_ids,
                    pixel_values=images,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=False
                )[0]
            elif args.model_architecture=="llama-3.2-vision":
                aspect_ratio_ids = batch["aspect_ratio_ids"]
                aspect_ratio_mask = batch["aspect_ratio_mask"]
                images = images.reshape(len(input_ids), -1, images.size(-4), images.size(-3), images.size(-2), images.size(-1))
                
                loss = model(
                    input_ids=input_ids,
                    pixel_values=images,
                    aspect_ratio_mask=aspect_ratio_mask,
                    aspect_ratio_ids=aspect_ratio_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=False)[0]

            model.backward(loss)
            model.step()
            
            acc_loss += loss
            acc_loss = get_all_reduce_mean(acc_loss).item()
            print_rank_0(f"Epoch {epoch+1}, Step: {step}, Loss:{acc_loss/(step+1)}", args.global_rank)

            global_step += 1

            if global_step % args.eval_step == 0:
                evaluation(model, eval_dataloader)

        model.tput_timer.update_epoch_count()
    
        # evaluation(model, eval_dataloader)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, f"epoch-{epoch}")
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                args.global_rank,
                                args.output_dir,
                                zero_stage=args.zero_stage, 
                                sub_folder=f"epoch-{epoch}")
        if args.zero_stage in [1,2]:
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
            model_to_save = model.module if hasattr(model,
                                                    'module') else model
            lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
            os.makedirs(f"{args.output_dir}/epoch-{epoch}", exist_ok=True)
            WEIGHTS_NAME = 'pytorch_model.bin'
            output_model_file = os.path.join(f'{args.output_dir}/epoch-{epoch}', WEIGHTS_NAME)
            torch.save(lean_state_dict, output_model_file)

if __name__ == '__main__':
    main()