#!/usr/bin/env python

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) + "/training")
from utils.data import build_dataset, DataCollatorPadToMaxLenForRewardModel, split_dataset, shuffle_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, fuse_lora, unfuse_lora
from utils.model import create_reward_or_critic_model

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a multi-modal task")

    parser.add_argument('--data_path',
                        type=str,
                        default='./data/',
                        help='Where the training data are stored.')
    
    parser.add_argument('--eval_data_path',
                        type=str,
                        default=None,
                        help='Where the evaluation data are stored.')

    parser.add_argument('--image_folder',
                    type=str,
                    default=None,
                    help='Where the image data are stored.')

    parser.add_argument('--data_debug_path',
                        type=str,
                        default=None,
                        help='If provided, will save 10 training samples'
                        'to the path for debug purpose.')

    parser.add_argument(
        "--data_train_split_ratio",
        type=float,
        default=0.9,
        help="Ratio of dataset to be splitted as train data. The remaining becomes eval data.",
    )
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
        default=8,
        help="The maximum number of images per sample.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="The maximum sequence length, note that image tokens are included.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_pretraining_components",
        type=float,
        default=0,
        help=
        "Initial learning rate for pre-trained weight, e.g., embedding (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=6,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=0,
        help="Number of steps (>1) or ratios (<=1) for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
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
        "--lm_reward_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--vision_reward_model_name_or_path", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--model_architecture", default="default", type=str)
    parser.add_argument("--reward_model_architecture", default="default", type=str)
    
    parser.add_argument("--reward_base_model", default="default", type=str)

    parser.add_argument(
        "--enable_mmca_attention",
        action='store_true',
        help="enable the new proposed attn, which is similar to cross attention",
    )
    parser.add_argument(
        "--vis_proj",
        type=str,
        default='baseline',
        help="[baseline, vit, or perceiver], used to projection vision feature to LLM embedding",
    )
    # deepspeed features
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help=
        "FP16 or BF16 precision. FP16 is recommended for typical use cases. BF16 is good for large models",
    )
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
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

    ## Listwise input setting
    parser.add_argument("--ranked_candidate_num",
                        type=int,
                        default=2,
                        help="Total number of candidate LLMs.")
    parser.add_argument('--template',
                    type=str,
                    choices=["default", "llama_2", "llama_3", "llama_3", "vicuna", "llava",
                            "llava_next", "llama-3.2-vision"],)
    parser.add_argument(
        '--from_checkpoint',
        type=str,
        default="./basemodel/",
        help='Specifying the checkpoint directory to be loaded.')
    parser.add_argument('--trained_reward_model',
                        type=str,
                        required=True,
                        help='Path to the trained reward model.')
    parser.add_argument(
        '--eval_step',
        type=int,
        default=100,
        help='The evaluation will be conducted every specific number of training steps.')
    parser.add_argument(
        '--save_step',
        type=int,
        default=999999,
        help='The checkpoint will be saved every specific number of training steps.')
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
    assert args.num_warmup_steps >= 0, "--num_warmup_steps must be >= 0"
    if 'qwen' in args.vision_reward_model_name_or_path.lower():
        assert args.vis_proj == 'baseline', "qwen's model only support baseline vis_proj as it has the perceiver module inside"
    
    args.reward_model_architecture = args.model_architecture
    args.reward_base_model = args.from_checkpoint

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

    ds_config = get_train_ds_config(args, offload=False,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()
    if args.model_architecture=="default":
        tokenizer = AutoTokenizer.from_pretrained(args.lm_reward_model_name_or_path,
                                                fast_tokenizer=True)

        tokenizer.padding_side = 'right'
    else:
        tokenizer = None
 
    model, image_processor, tokenizer = create_reward_or_critic_model(
            text_tokenizer=tokenizer,
            ds_config=ds_config,
            is_reward=True,
            training_reward_stage=True,
            args=args)
    
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    if os.path.exists(args.trained_reward_model):
        model.load_state_dict(torch.load(os.path.join(args.trained_reward_model, 'pytorch_model.bin'), map_location='cpu'), strict=False)

    if args.lang_lora_dim > 0:
        model.lang_decoder = convert_linear_layer_to_lora(model.rwtranrsformer.lang_decoder, args.lang_lora_module_name, args.lang_lora_dim)
        if args.only_optimize_lora:
            model.lang_decoder = only_optimize_lora_parameters(model.rwtranrsformer.lang_decoder)

    if args.vis_lora_dim > 0:
        model.vis_encoder = convert_linear_layer_to_lora(model.rwtranrsformer.vis_encoder, args.vis_lora_module_name, args.vis_lora_dim)
        if args.only_optimize_lora:
            model.vis_encoder = only_optimize_lora_parameters(model.rwtranrsformer.vis_encoder)

    print_rank_0(model, args.global_rank) 
        
    # Prepare the data
    if len(args.dataset_samples) < len(args.dataset_names):
        assert len(args.dataset_samples) == 1, "when args.dataset_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_samples =  [args.dataset_samples[0]] * len(args.dataset_names)
    if len(args.dataset_concatenate_samples) < len(args.dataset_names):
        assert len(args.dataset_concatenate_samples) == 1, "when args.dataset_concatenate_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_concatenate_samples =  [args.dataset_concatenate_samples[0]] * len(args.dataset_names)
    # convert to int
    args.dataset_concatenate_samples = [int(i) for i in args.dataset_concatenate_samples]

    if args.eval_data_path == None:       
        dataset = build_dataset(
            args.data_path,
            args.data_debug_path,
            args.dataset_names,
            args.dataset_samples,
            args.dataset_concatenate_samples,
            args.max_num_image_per_sample,
            max_ranked_candidate_num=args.ranked_candidate_num,
            vis_processor=image_processor,
            vis_root=args.image_folder,
            tokenizer=tokenizer,
            template=args.template
        )
        # split the dataset into train and evaluation
        np_rng = np.random.RandomState(seed=args.seed)
        dataset = shuffle_dataset(dataset, np_rng)
        train_dataset, eval_dataset = split_dataset(dataset, args.data_train_split_ratio)
    else:
        train_dataset = build_dataset(
            args.data_path,
            args.data_debug_path,
            args.dataset_names,
            args.dataset_samples,
            args.dataset_concatenate_samples,
            args.max_num_image_per_sample,
            max_ranked_candidate_num=args.ranked_candidate_num,
            vis_processor=image_processor,
            vis_root=args.image_folder,
            tokenizer=tokenizer,
            template=args.template
        )
        eval_dataset = build_dataset(
            args.eval_data_path,
            args.data_debug_path,
            args.dataset_names,
            args.dataset_samples,
            args.dataset_concatenate_samples,
            args.max_num_image_per_sample,
            max_ranked_candidate_num=args.ranked_candidate_num,
            vis_processor=image_processor,
            vis_root=args.image_folder,
            tokenizer=tokenizer,
            template=args.template
        )
    
    if args.model_architecture == "llama-3.2-vision":
        image_size = image_processor.size
    else:
        image_size = image_processor.crop_size

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
        collate_fn=DataCollatorPadToMaxLenForRewardModel(args.max_seq_len, tokenizer.pad_token_id, image_size),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=DistributedSampler(eval_dataset, shuffle=False),
        collate_fn=DataCollatorPadToMaxLenForRewardModel(args.max_seq_len, tokenizer.pad_token_id, image_size),
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
        print_rank_0("***** Evaluation Begin *****")
        model.eval()
        batch_size = args.per_device_eval_batch_size
        acc_count = 0
        total = 0
        candidate_assigned = False
        candidate_size = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                batch = to_device(batch, device)
                chosen_idx = [i for i in range(len(batch["input_ids"])) if (batch['input_ids'][i][0] != -1)]

                batch["image"] = [batch["image"][i] for i in chosen_idx]
                batch["input_ids"] = [batch["input_ids"][i] for i in chosen_idx]
                batch["attention_mask"] = [batch["attention_mask"][i] for i in chosen_idx]
                batch["labels"] = [batch["labels"][i] for i in chosen_idx]
                
                input_ids = torch.stack(batch["input_ids"])
                attention_mask = torch.stack(batch["attention_mask"])
                
                labels = torch.stack(batch["labels"])
                images = torch.stack(batch["image"])

                if args.model_architecture == "llava_next":
                    image_sizes = batch["image_sizes"]
                    image_sizes = image_sizes.reshape(len(input_ids), 2)
                    images = images.reshape(len(input_ids), 5, images.size(-3), images.size(-2), images.size(-1))
                    aspect_ratio_ids = None
                    aspect_ratio_mask = None

                elif args.model_architecture == "llama-3.2-vision":
                    aspect_ratio_ids = batch["aspect_ratio_ids"]
                    aspect_ratio_mask = batch["aspect_ratio_mask"]
                    images = images.reshape(len(input_ids), 1, images.size(-4), images.size(-3), images.size(-2), images.size(-1))
                   
                    image_sizes = None

                else:
                    
                    image_sizes = None
                    aspect_ratio_ids = None
                    aspect_ratio_mask = None

                labels_tmp = input_ids.clone()
                attention_mask_tmp = attention_mask.clone()
                attention_mask_tmp[attention_mask_tmp==0] = 1


                reward_scores = model(
                    images ,
                    input_ids,
                    image_sizes=image_sizes,
                    aspect_ratio_ids=aspect_ratio_ids,
                    aspect_ratio_mask=aspect_ratio_mask,
                    attention_mask=attention_mask_tmp,
                    input_labels=labels_tmp,
                    image_num=batch["image_num"],
                )

                if candidate_assigned == False:
                    candidate_size = int(len(reward_scores) / batch_size)
                    candidate_assigned = True
                for i in range(int(len(reward_scores) / candidate_size)):
                    for j in range(0, candidate_size-1):
                        for k in range(j+1, candidate_size):
                            if reward_scores[i*candidate_size + j] > reward_scores[i*candidate_size + k]:
                                acc_count = acc_count + 1
                            total = total + 1
        model.train()
        acc_rate = acc_count / total
        try:
            acc_rate = get_all_reduce_mean(acc_rate).item()
        except:
            pass
        print_rank_0(
            f"Eval accuracy: {acc_rate}, Avg of Reward Scores: {acc_rate}", 
            args.global_rank)
        return acc_rate
    
    # Train!
    if start_epoch == 0:
        print_rank_0("***** Before training *****", args.global_rank)
        evaluation(model, eval_dataloader)

    print_rank_0("***** Running training *****", args.global_rank)
    for epoch in range(start_epoch, args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        acc_loss = 0
        model.train()

        global_step = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # batch--> y1 of sample 1; y2 of sample 1;...; yn of sample 1; y1 of sample 2; ...
            batch = to_device(batch, device)  #torch.size(1, 3, 224, 224]) #torch.Size([1, 1, 3, 224, 224])
            chosen_idx = [i for i in range(len(batch["input_ids"])) if (batch['input_ids'][i][0] != -1)]

            batch["image"] = [batch["image"][i] for i in chosen_idx]
            batch["input_ids"] = [batch["input_ids"][i] for i in chosen_idx]
            batch["attention_mask"] = [batch["attention_mask"][i] for i in chosen_idx]
            batch["labels"] = [batch["labels"][i] for i in chosen_idx]

            input_ids = torch.stack(batch["input_ids"])
            attention_mask = torch.stack(batch["attention_mask"])
            labels = torch.stack(batch["labels"])
            images = torch.stack(batch["image"])

            if args.model_architecture == "llava_next":
                image_sizes = batch["image_sizes"]
                image_sizes = image_sizes.reshape(len(input_ids), 2)
                images = images.reshape(len(input_ids), 5, images.size(-3), images.size(-2), images.size(-1))
                aspect_ratio_ids = None
                aspect_ratio_mask = None

            elif args.model_architecture == "llama-3.2-vision":
                aspect_ratio_ids = batch["aspect_ratio_ids"]
                aspect_ratio_mask = batch["aspect_ratio_mask"]
                images = images.reshape(len(input_ids), 1, images.size(-4), images.size(-3), images.size(-2), images.size(-1))
                image_sizes = None

            else:
                image_sizes = None
                aspect_ratio_ids = None
                aspect_ratio_mask = None
            
            labels_tmp = input_ids.clone()
            attention_mask_tmp = attention_mask.clone()
            attention_mask_tmp[attention_mask_tmp==0] = 1

            reward_scores = model(
                images,
                input_ids,
                image_sizes=image_sizes,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                attention_mask=attention_mask_tmp,
                input_labels=labels_tmp,
                image_num=batch["image_num"],
            )

            # reward modeling
            # using Plackett-Luce to compute loss
            all_loss = []
            all_comparison_judgement = [] # compute correct predictions

            sample_num = len(reward_scores)//args.ranked_candidate_num
            count_list = [args.ranked_candidate_num] * sample_num

            for batch_index, can_num in enumerate(count_list):
                for i in range(can_num):
                    numerator_reward_scores = reward_scores[sum(count_list[:batch_index]) + i]
                    
                    denominator_reward_scores_sum = []
                    for j in range(i, can_num):
                        denominator_reward_score = reward_scores[sum(count_list[:batch_index]) + j]
                        denominator_reward_scores_sum.append(torch.exp(denominator_reward_score))

                    # get comparison
                    for k in range(i+1, can_num):
                        if reward_scores[sum(count_list[:batch_index]) + i] > reward_scores[sum(count_list[:batch_index]) + k]:
                            all_comparison_judgement.append(True)
                        else:
                            all_comparison_judgement.append(False)
                    
                    all_loss.append(-torch.log(torch.exp(numerator_reward_scores) /
                                                    torch.stack(denominator_reward_scores_sum, dim=0).sum(0)))

            loss = torch.stack(all_loss, dim=0).sum(0) / len(count_list)

            model.backward(loss)
            model.step()

            acc_loss += loss
            acc_loss = get_all_reduce_mean(acc_loss).item()

            print_rank_0(
                f'Epoch {epoch+1}, Step: {(step+1)}, Loss: {acc_loss/(step+1)}, '+ \
                f'Accuracy: {sum(all_comparison_judgement)/len(all_comparison_judgement)}',
                args.global_rank)
            
            global_step += 1
            if global_step % args.eval_step == 0:
                evaluation(model, eval_dataloader)

            if global_step % args.save_step == 0:
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, f'epoch-{epoch}-{global_step}')

                if args.zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                    save_zero_three_model(model,
                                        args.global_rank,
                                        args.output_dir,
                                        zero_stage=args.zero_stage, 
                                        sub_folder=f'epoch-{epoch}-{global_step}')
                if args.zero_stage in [1,2]:
                    # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model
                    lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                    os.makedirs(f'{args.output_dir}/epoch-{epoch}-{global_step}', exist_ok=True)
                    WEIGHTS_NAME = "pytorch_model.bin"
                    output_model_file = os.path.join(f'{args.output_dir}/epoch-{epoch}-{global_step}', WEIGHTS_NAME)
                    torch.save(lean_state_dict, output_model_file)

        model.tput_timer.update_epoch_count()
        evaluation(model, eval_dataloader)

        model = fuse_lora(model)
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, f'epoch-{epoch}')
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                args.global_rank,
                                args.output_dir,
                                zero_stage=args.zero_stage, 
                                sub_folder=f'epoch-{epoch}')
        if args.zero_stage in [1,2]:
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
            model_to_save = model.module if hasattr(model,
                                                    'module') else model
            lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
            os.makedirs(f'{args.output_dir}/epoch-{epoch}', exist_ok=True)
            WEIGHTS_NAME = "pytorch_model.bin"
            output_model_file = os.path.join(f'{args.output_dir}/epoch-{epoch}', WEIGHTS_NAME)
            torch.save(lean_state_dict, output_model_file)

if __name__ == "__main__":
    main()