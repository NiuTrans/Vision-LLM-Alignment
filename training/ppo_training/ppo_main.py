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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from transformers import (
    SchedulerType,
    get_scheduler,
    AutoTokenizer
)

import deepspeed
from rlhf_engine import DeepSpeedRLHFEngine
from ppo_training_utils import (sampling, compute_logprobs_from_actor_and_ref, 
    compute_kl_reward_scores, get_advantages_and_returns, 
    critic_loss_fn, gather_log_probs,
    actor_loss_fn)

from transformers import AdamW
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data import build_dataset, DataCollatorPadToMaxLenForPPOTraining, split_dataset, shuffle_dataset, DST
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, fuse_lora, unfuse_lora
from utils.model import create_dsvl_model_and_transforms

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a multi-modal task")

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
        default=1,
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
        "--lm_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--vision_model_name_or_path", default="openai/clip-vit-large-patch14", type=str)
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

    ## from ppo training
    parser.add_argument('--sft_model_ckpt_path',
                        type=str,
                        required=True,
                        help='Path to the trained SFT model.')
    parser.add_argument('--reward_model_ckpt_path',
                        type=str,
                        required=True,
                        help='Path to the trained reward model.')
    parser.add_argument(
        "--lm_reward_model_name_or_path",
        type=str,
        help=
        "Path to the used pretrained model for training reward models.",
        required=True)
    parser.add_argument(
        "--vision_reward_model_name_or_path", 
        default="openai/clip-vit-large-patch14", 
        help=
        "Path to the used vision model for training reward models.",
        type=str)
    # Here we do not set the reference model and critic model.
    # We use the sft model to initialize the reference model, 
    # and use the reward model to initialize the critic model.
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and reference).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--offload_actor_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for actor and reference model.')
    parser.add_argument(
        '--offload_critic_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for critic and reward model.')
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=3e-5,
        help=
        "Initial actor learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=3e-6,
        help=
        "Initial critic learning rate (after the potential warmup period) to use.",
    )
    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    
    parser.add_argument(
        '--max_training_samples_num',
        type=int,
        default=10000,
        help='The maximum number of training samples in the PPO process.')
    
    parser.add_argument(
        '--save_step',
        type=int,
        default=1000,
        help='A checkpoint is saved every specific number of training steps.')

    parser.add_argument(
        '--eval_step',
        type=int,
        default=100,
        help='The evaluation will be conducted every specific number of training steps.')

    parser.add_argument('--template',
                type=str,
                choices=["default", "llama_2", "llama_3", "llama_3", "vicuna"],)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.learning_rate_pretraining_components == 0.0:
        # if we do not provide special learning rate, mainly for embedding, the same lr is applied
        args.learning_rate_pretraining_components = args.actor_learning_rate
    assert args.num_warmup_steps >= 0, "--num_warmup_steps must be >= 0"
    if 'qwen' in args.vision_model_name_or_path.lower():
        assert args.vis_proj == 'baseline', "qwen's model only support baseline vis_proj as it has the perceiver module inside"
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
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.padding_side = 'left'

    reward_tokenizer = AutoTokenizer.from_pretrained(args.lm_reward_model_name_or_path,
                                              fast_tokenizer=True)
    reward_tokenizer.padding_side = 'right'
    
    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.sft_model_ckpt_path,
        reward_model_name_or_path=args.reward_model_ckpt_path,
        actor_tokenizer=tokenizer,
        reward_tokenizer=reward_tokenizer,
        number_dataset=args.max_training_samples_num,
        args=args)
        
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
        vis_processor=rlhf_engine.actor_image_processor,
        vis_root=args.image_folder,
        tokenizer=rlhf_engine.actor_tokenizer_new,
        template=args.template
    )

    # split the dataset into train and evaluation
    total_data = len(dataset)
    np_rng = np.random.RandomState(seed=args.seed)
    dataset = shuffle_dataset(dataset, np_rng)
    train_dataset, eval_dataset = split_dataset(dataset, args.data_train_split_ratio)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
        collate_fn=DataCollatorPadToMaxLenForPPOTraining(args.max_seq_len, tokenizer.pad_token_id),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=DistributedSampler(eval_dataset, shuffle=False),
        collate_fn=DataCollatorPadToMaxLenForPPOTraining(args.max_seq_len, tokenizer.pad_token_id),
    )

    start_epoch = 0

    if args.gradient_checkpointing:
        rlhf_engine.actor.gradient_checkpointing_enable()
        rlhf_engine.critic.gradient_checkpointing_enable()

    def evaluation(model, eval_dataloader):
        print_rank_0("***** Running training *****", args.global_rank)
        model.eval()
        reward_score_acc = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = to_device(batch, device)  #torch.size(1, 3, 224, 224]) #torch.Size([1, 1, 3, 224, 224])
            images = batch["image"].half() 
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            with torch.no_grad():
                # generation
                sampling_ans = sampling(rlhf_engine.actor, 
                                images, input_ids, 
                                attention_mask=attention_mask, 
                                input_labels=labels, 
                                max_seq_len=args.max_seq_len,
                                pad_token_id=rlhf_engine.actor_tokenizer_new.pad_token_id)
            
            # compute reward scores
            question_strings = rlhf_engine.actor_tokenizer_new.batch_decode(input_ids)
            question_answer_pairs_strinig = [q.replace(rlhf_engine.actor_tokenizer_new.bos_token, "").replace(rlhf_engine.actor_tokenizer_new.pad_token, "").strip(" ") + \
                                            a[1] for q, a in zip(question_strings, sampling_ans)]
            # encoder with reward model's tokenizer
            question_answer_pairs = rlhf_engine.critic_tokenizer_new(question_answer_pairs_strinig, 
                                                                    padding=True, 
                                                                    add_special_tokens=True,
                                                                    return_tensors="pt")

            question_answer_pairs = to_device(question_answer_pairs, device)
            reward_input_id = question_answer_pairs["input_ids"]
            reward_attention_mask = question_answer_pairs["attention_mask"]

            rlhf_engine.critic_tokenizer_new.decode(input_ids[0])

            with torch.no_grad():
                reward_scores = rlhf_engine.reward.forward_value(images,
                                    reward_input_id,
                                    attention_mask=reward_attention_mask,
                                    image_num=batch["image_num"]
                                )["chosen_end_scores"]


            reward_score_acc += reward_scores.mean()
        model.train()
        reward_score_acc = get_all_reduce_mean(reward_score_acc).item()
        reward_score_avg = reward_score_acc/(step+1)
        print_rank_0(f"the eval average reward scores: {reward_score_avg}", args.global_rank)
        return reward_score_avg
    
    # Train!
    if start_epoch == 0:
        print_rank_0("***** Before training *****", args.global_rank)
        evaluation(rlhf_engine.actor, eval_dataloader)

    print_rank_0("***** Running training *****", args.global_rank)
    global_step = 0
    for epoch in range(start_epoch, args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)

        rlhf_engine.actor.train()
        rlhf_engine.critic.train()
        rlhf_engine.ref.eval()
        rlhf_engine.reward.eval()

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = to_device(batch, device)  #torch.size(1, 3, 224, 224]) #torch.Size([1, 1, 3, 224, 224])
            images = batch["image"].half() 
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            # Step 1: sampling candidate answers
            sampling_ans = sampling(rlhf_engine.actor, 
                                    images, input_ids, 
                                    attention_mask=attention_mask, 
                                    input_labels=labels, 
                                    max_seq_len=args.max_seq_len,
                                    pad_token_id=rlhf_engine.actor_tokenizer_new.pad_token_id)
            
            # Step 2: computing reward scores
            # Firstly, we convert the question to a string format. 
            # We then concat the question string with the sampled answer string.
            # question_strings = rlhf_engine.actor_tokenizer_new.
            question_strings = rlhf_engine.actor_tokenizer_new.batch_decode(input_ids)
            question_answer_pairs_strinig = [q.replace(rlhf_engine.actor_tokenizer_new.bos_token, "").replace(rlhf_engine.actor_tokenizer_new.pad_token, "").strip(" ") + \
                                            a[1] for q, a in zip(question_strings, sampling_ans)]
            
            # encoder with reward model's tokenizer
            question_answer_pairs = rlhf_engine.critic_tokenizer_new(question_answer_pairs_strinig, 
                                                                    padding=True, 
                                                                    add_special_tokens=True,
                                                                    return_tensors="pt")

            question_answer_pairs = to_device(question_answer_pairs, device)
            reward_input_id = question_answer_pairs["input_ids"]
            reward_attention_mask = question_answer_pairs["attention_mask"]

            rlhf_engine.critic_tokenizer_new.decode(input_ids[0])

            with torch.no_grad():
                reward_scores = rlhf_engine.reward.forward_value(images,
                                    reward_input_id,
                                    attention_mask=reward_attention_mask,
                                    image_num=batch["image_num"]
                                )["chosen_end_scores"]

            # Step 3: computing KL and values of the crtic model
            # computing the label ids for the critic model
            critic_ids = [torch.cat((input_ids[index], sampling_ans[index][0][0]), dim=-1)
                for index in range(len(input_ids))]
            # pading thie critic_ids
            critic_input_ids = pad_sequence([ids for ids in critic_ids], 
                                        padding_value=rlhf_engine.actor_tokenizer_new.pad_token_id, 
                                        batch_first=True)

            label_ids = []
            for index in range(reward_input_id.size()[0]):
                sub_labels = [DST.DEFAULT_LABEL_PADDING_NUM] * len(input_ids[index]) + \
                                critic_input_ids[index][len(input_ids[index]):].tolist()
                
                label_ids.append(sub_labels)

            critic_label_ids = torch.LongTensor(label_ids).cuda()

            critic_attention_mask = critic_input_ids.not_equal(rlhf_engine.actor_tokenizer_new.pad_token_id).long()

            action_attention_mask = critic_attention_mask[:, 1:]

            # compute logprobs and ref_logprobs
            logprobs, ref_logprobs = compute_logprobs_from_actor_and_ref(actor_model=rlhf_engine.actor,
                                    ref_model=rlhf_engine.ref,
                                    images=images,
                                    input_ids=critic_input_ids,
                                    input_labels=critic_label_ids,
                                    attention_mask=critic_attention_mask,
                                    image_num=batch["image_num"])
            
            # Step 4: compute advantages and returns
            # compute the values
            with torch.no_grad():
                old_values = rlhf_engine.critic.forward_value(images,
                                                critic_input_ids,
                                                attention_mask=critic_attention_mask,
                                                image_num=batch["image_num"]
                                            )["values"]
            
            # run ppo training. 
            # Note that we first implement the case of a minibatch equal to the training batch.
            actor_loss_log = 0
            critic_loss_log = 0
            kl_distance_log = 0
            for ppo_ep in range(args.ppo_epochs):
                if ppo_ep != 0:
                    with torch.no_grad():
                        actor_logits = rlhf_engine.actor(images,
                                        critic_input_ids,
                                        attention_mask=critic_attention_mask,
                                        input_labels=critic_label_ids,
                                        image_num=batch["image_num"]
                                        )[1]
                        logprobs = gather_log_probs(actor_logits[:, :-1, :], critic_input_ids[:, 1:])

                # compute reward scores with KL
                start = input_ids.shape[1] - 1
                kl_reward_scores, kl_distance = compute_kl_reward_scores(logprobs=logprobs,
                                            ref_logprobs=ref_logprobs,
                                            reward_scores=reward_scores,
                                            start=start,
                                            attention_mask=action_attention_mask)

                # we need to zero out the reward and value after the end of the conversation
                # otherwise the advantage/return will be wrong
                ends = start + action_attention_mask[:, start:].sum(1) + 1

                for i in range(kl_reward_scores.shape[0]):
                    kl_reward_scores[i, ends[i]:] = 0
                    old_values[i, ends[i]:] = 0

                advantages, returns = get_advantages_and_returns(values=old_values, 
                                                    rewards=kl_reward_scores, 
                                                    start=start)
                
                # Step 5: update the actor and critic models
                # update critic model
                values = rlhf_engine.critic.forward_value(images,
                                                    critic_input_ids,
                                                    attention_mask=critic_attention_mask,
                                                    image_num=batch["image_num"]
                                                )["values"]

                critic_loss = critic_loss_fn(values=values[:, start:], 
                                            old_values=old_values[:,start:],
                                            returns=returns, 
                                            mask=action_attention_mask[:, start:])

                rlhf_engine.critic.backward(critic_loss)

                # update actor model
                actor_logits = rlhf_engine.actor(images,
                                        critic_input_ids,
                                        attention_mask=critic_attention_mask,
                                        input_labels=critic_label_ids,
                                        image_num=batch["image_num"]
                                        )[1]
                actor_logprobs = gather_log_probs(actor_logits[:, :-1, :], critic_input_ids[:, 1:])
                actor_loss = actor_loss_fn(logprobs=actor_logprobs[:, start:],
                                        old_logprobs=logprobs[:, start:], 
                                        advantages=advantages,
                                        mask=action_attention_mask[:, start:])

                rlhf_engine.actor.backward(actor_loss)

                if not args.align_overflow:
                    rlhf_engine.actor.step()

                if args.align_overflow:
                    actor_overflow = rlhf_engine.actor.optimizer.check_overflow(
                        external=True)
                    critic_overflow = rlhf_engine.critic.optimizer.check_overflow(
                        external=True)

                    rank = torch.distributed.get_rank()
                    if actor_overflow and not critic_overflow:
                        rlhf_engine.critic.optimizer.skip_step = True
                        print_rank_0(
                            "OVERFLOW: actor overflow, skipping both actor and critic steps",
                            rank)
                    elif not actor_overflow and critic_overflow:
                        rlhf_engine.actor.optimizer.skip_step = True
                        print_rank_0(
                            "OVERFLOW: critic overflow, skipping both actor and critic steps",
                            rank)
                    elif actor_overflow and critic_overflow:
                        print_rank_0(
                            "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                            rank)
                    rlhf_engine.actor.step()

                rlhf_engine.critic.step()
            
                actor_loss_log += actor_loss
                critic_loss_log += critic_loss
                kl_distance_log += kl_distance

                actor_loss_log = get_all_reduce_mean(actor_loss_log).item()
                critic_loss_log = get_all_reduce_mean(critic_loss_log).item()
                kl_distance_log = get_all_reduce_mean(kl_distance_log).item()

            print_rank_0(
                f'Epoch {epoch+1}, Step: {step+1}, Actor Loss:{actor_loss_log/args.ppo_epochs},'+ \
                f'Critic Loss:{critic_loss_log/args.ppo_epochs}, Reward Score: {reward_scores.mean()},'+ \
                f'KL Distance: {kl_distance_log/args.ppo_epochs}', 
                args.global_rank)

            global_step += 1
            if global_step % args.save_step == 0:

                model = fuse_lora(rlhf_engine.actor)
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, f'{args.output_dir}/epoch-{epoch}-step-{global_step}')
                if args.actor_zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                    save_zero_three_model(model,
                                        args.global_rank,
                                        args.output_dir,
                                        zero_stage=args.zero_stage, 
                                        sub_folder=f'epoch-{epoch}')
                if args.actor_zero_stage in [1,2]:
                    # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model
                    lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                    os.makedirs(f'{args.output_dir}/epoch-{epoch}-step-{global_step}', exist_ok=True)
                    WEIGHTS_NAME = "pytorch_model.bin"
                    output_model_file = os.path.join(f'{args.output_dir}/epoch-{epoch}-step-{global_step}', WEIGHTS_NAME)
                    torch.save(lean_state_dict, output_model_file)
            
            if global_step % args.eval_step == 0:
                evaluation(rlhf_engine.actor, eval_dataloader)

        evaluation(rlhf_engine.actor, eval_dataloader)

        model = fuse_lora(rlhf_engine.actor)
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, f'{args.output_dir}/epoch-{epoch}')
        if args.actor_zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                args.global_rank,
                                args.output_dir,
                                zero_stage=args.zero_stage, 
                                sub_folder=f'epoch-{epoch}')
        if args.actor_zero_stage in [1,2]:
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