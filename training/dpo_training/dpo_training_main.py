#!/usr/bin/env python

import argparse
import os
import math
import sys
import numpy as np

import random
import torch.distributed
from tqdm import tqdm
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
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
from utils.data import build_dataset, DataCollatorPadToMaxLenForRewardModel, split_dataset, shuffle_dataset, DST
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
    parser.add_argument("--beta",
                        type=float,
                        default=1e-1,
                        help="Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0."

    )
    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.0,
                        help="conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)"
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
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
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
                        default=1235,
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

    ## Listwise input setting
    parser.add_argument("--ranked_candidate_num",
                        type=int,
                        default=2,
                        help="Total number of candidate LLMs.")
    parser.add_argument(
        '--from_checnkpoint',
        type=str,
        default="./basemodel/",
        help='Specifying the checkpoint directory to be loaded.')
    parser.add_argument('--template',
                        type=str,
                        choices=["default", "llama_2", "llama_3", "llama_3", "vicuna"],)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.learning_rate_pretraining_components == 0.0:
        # if we do not provide special learning rate, mainly for embedding, the same lr is applied
        args.learning_rate_pretraining_components = args.learning_rate
    assert args.num_warmup_steps >= 0, "--num_warmup_steps must be >= 0"
    if 'qwen' in args.vision_model_name_or_path.lower():
        assert args.vis_proj == 'baseline', "qwen's model only support baseline vis_proj as it has the perceiver module inside"
    return args

def gather_log_probs(logits, labels, label_mask):
    label_mask_new = (label_mask.clone() != DST.DEFAULT_LABEL_PADDING_NUM).int()[:,1:]
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))

    return (log_probs_labels.squeeze(-1)*label_mask_new).sum(-1)

def main():

    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    
    args.global_rank = torch.distributed.get_rank()

    set_random_seed(args.seed)

    torch.distributed.barrier()
    tokenizer_origin = AutoTokenizer.from_pretrained(
        args.lm_model_name_or_path,
        fast_tokenizer = True
    )
    tokenizer_origin.padding_side = 'left'

    ds_config = get_train_ds_config(args, offload=False, stage=2)

    model, image_processor, tokenizer = create_dsvl_model_and_transforms(
        text_tokenizer=tokenizer_origin,
        args=args,
        ds_config=ds_config
    )

    if args.lang_lora_dim > 0:
        model.lang_decoder = convert_linear_layer_to_lora(
            model.lang_decoder,
            args.lang_lora_module_name,
            args.lang_lora_dim
        )
        if args.only_optimize_lora:
            model.lang_decoder = only_optimize_lora_parameters(
                model.lang_decoder
            )

    if args.vis_lora_dim > 0:
        model.vis_encoder = convert_linear_layer_to_lora(
            model.vis_encoder, 
            args.vis_lora_module_name,
            args.vis_lora_dim
        )
        if args.only_optimize_lora:
            model.vis_encoder = only_optimize_lora_parameters(
                model.vis_encoder
            )
    print_rank_0(model, args.global_rank)

    # prepare the data
    if len(args.dataset_samples) < len(args.dataset_names):
        assert len(args.dataset_samples) == 1, "when args.dataset_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_samples =  [args.dataset_samples[0]] * len(args.dataset_names)
    if len(args.dataset_concatenate_samples) < len(args.dataset_names):
        assert len(args.dataset_concatenate_samples) == 1, "when args.dataset_concatenate_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_concatenate_samples =  [args.dataset_concatenate_samples[0]] * len(args.dataset_names)

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

    np_rng = np.random.RandomState(seed=args.seed)
    dataset = shuffle_dataset(dataset, np_rng)
    train_dataset = dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
        collate_fn=DataCollatorPadToMaxLenForRewardModel(args.max_seq_len, tokenizer.pad_token_id),
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

    print_rank_0("load actor model............")
    model.load_state_dict(torch.load(os.path.join(args.from_checnkpoint, 'pytorch_model.bin'), map_location='cpu'), strict=False)

    ds_config = get_train_ds_config(args, offload=args.offload,
                                    stage=args.zero_stage)
    
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size

    ds_config['train_batch_size'] = args.per_device_train_batch_size * \
                                    torch.distributed.get_world_size() * args.gradient_accumulation_steps

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
    
    # ref model prepare
    ds_ref_config_load = get_train_ds_config(
        offload=args.offload,
        args=args,
        stage=2 
    )

    ref_model , _, _ = create_dsvl_model_and_transforms(
        text_tokenizer=tokenizer_origin,
        args=args,
        ds_config=ds_ref_config_load
    )

    if args.lang_lora_dim > 0:
        ref_model.lang_encoder = convert_linear_layer_to_lora(
            ref_model.lang_decoder,
            args.lang_lora_module_name,
            args.lang_lora_dim
        )
        if args.only_optimize_lora:
            ref_model.lang_decoder = only_optimize_lora_parameters(
                ref_model.lang_decoder
            )
    
    if args.vis_lora_dim > 0:
        ref_model.vis_encoder = convert_linear_layer_to_lora(
            ref_model.vis_encoder,
            args.vis_lora_module_name,
            args.vis_lora_dim
        )
        if args.only_optimize_lora:
            ref_model.vis_encoder = only_optimize_lora_parameters(
                ref_model.vis_encoder
            )

    print_rank_0("load ref model............")
    ref_model.load_state_dict(torch.load(os.path.join(args.from_checnkpoint, 'pytorch_model.bin'), map_location='cpu'), strict=False)

    ds_ref_config_training = get_train_ds_config(
        offload=args.offload,
        args=args,
        stage=3
    )

    ref_model, *_ = deepspeed.initialize(
        model=ref_model,
        config=ds_ref_config_training
    )

    print_rank_0("***** Running training *****", args.global_rank)
    for epoch in range(start_epoch, args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        dpo_training_loss = 0
        global_step = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # batch--> y1 of sample 1; y2 of sample 1;...; yn of sample 1; y1 of sample 2; ...
            batch = to_device(batch, device)  #torch.size(1, 3, 224, 224]) #torch.Size([1, 1, 3, 224, 224])
            images = batch["image"].half() 
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs_logits = model(images,
                input_ids,
                attention_mask=attention_mask,
                input_labels=labels,
                image_num=batch["image_num"])[1]

            with torch.no_grad():
                ref_outputs_logits = ref_model(
                    images,
                    input_ids,
                    attention_mask=attention_mask,
                    input_labels=labels,
                    image_num=batch["image_num"],
                )[1]
            
            logprobs = gather_log_probs(outputs_logits[:, :-1, :], input_ids[:, 1:], labels)
            ref_logprobs = gather_log_probs(ref_outputs_logits[:, :-1, :], input_ids[:,1:], labels)
            
            if logprobs.shape[0] != args.per_device_train_batch_size * 2:
                print("check the number of the candidate, this dpo training only supports two, chosen and rejected !!!")
                outputs_logits = outputs_logits.detach()
                continue
            
            sample_num = len(logprobs) // 2
            loss = 0
            for batch_index in range(sample_num):
                chosen_logps = logprobs[batch_index*2]
                rejected_logps = logprobs[batch_index*2 + 1]

                ref_chosen_logps = ref_logprobs[batch_index*2]
                ref_rejected_logps = ref_logprobs[batch_index*2 + 1]

                #compute loss
                logits = args.beta * ((chosen_logps-ref_chosen_logps)-(rejected_logps-ref_rejected_logps))
                loss += (-torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) - \
                            torch.nn.functional.logsigmoid(-logits) * args.label_smoothing)
            
            loss = loss/sample_num 
            model.backward(loss)
            model.step()

            dpo_training_loss += loss

            dpo_training_loss = get_all_reduce_mean(dpo_training_loss).item()
            global_step += 1
            print_rank_0(f'Epoch {epoch}, Step: {(step)}, Loss:{dpo_training_loss/global_step}', args.global_rank)
        
        model = fuse_lora(model)
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, f'{args.output_dir}/epoch-{epoch}')
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

