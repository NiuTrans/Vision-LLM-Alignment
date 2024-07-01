#!/bin/bash

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=base_models/vision_encoder/clip-vit-large-patch14
LLM=base_models/llama-3-8b-Instruct

sft_model_ckpt_path=models/sft_test/epoch-3
reward_model_ckpt_path=models/reward_test/epoch-3

TEMPLATE=llama_3

lm_reward_model_name_or_path=$LLM
vision_reward_model_name_or_path=$VISION_MODEL


actor_zero_stage=2
critic_zero_stage=3

ACTOR_LEARNING_RATE=3e-5
CRITIC_LEARNING_RATE=3e-6

EPOCH=3
ZERO_STAGE=2
lr=1e-3

IMAGE_FOLDER=data/coco_2017/
DATA_PATH=data/ppo_samples.json
TRAIN_SPLIT_RATIO=0.95

DATA="llava_ppo"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"

OUTPUT=models/ppo_test

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
deepspeed --include localhost:0,1,2,3 --master_port 12346 training/ppo_training/ppo_main.py --max_seq_len 1024 \
    --data_path ${DATA_PATH} --image_folder ${IMAGE_FOLDER} --template ${TEMPLATE} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --data_train_split_ratio ${TRAIN_SPLIT_RATIO} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 8 \
    --lm_reward_model_name_or_path  ${LLM} \
    --vision_reward_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 4 --num_warmup_steps 0.1 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --deepspeed --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} --ppo_epochs 4 --enable_mmca_attention \
    --precision bf16 \
    --sft_model_ckpt_path $sft_model_ckpt_path \
    --reward_model_ckpt_path $reward_model_ckpt_path \
    --lm_model_name_or_path $LLM \
    --vision_model_name_or_path $VISION_MODEL \
    --lm_reward_model_name_or_path $lm_reward_model_name_or_path \
    --vision_reward_model_name_or_path $vision_reward_model_name_or_path \
    --actor_zero_stage $actor_zero_stage --critic_zero_stage $critic_zero_stage \
    --image_folder /localnvme/application/sc_new/wangchenglong_56/rlhf_llama_vision/data/coco2017_flickr30k_comb \
    --actor_learning_rate $ACTOR_LEARNING_RATE --critic_learning_rate $CRITIC_LEARNING_RATE 
