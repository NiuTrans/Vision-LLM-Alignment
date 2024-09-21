#!/bin/bash

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=base_models/vision_encoder/clip-vit-large-patch14
LLM=/mnt/wangchenglong/ganyang/models/LLaVA_MORE-llama_3_1-8B-S2-finetuning

# FROM_CHECKPOINT=models/sft_test/epoch-3

TEMPLATE=llama_3
EPOCH=3
ZERO_STAGE=3

lr=3e-5

DATA_PATH=data/comparsion_reward_dpo.json
# EVAL_DATA_PATH=data/reward_samples_test.json
CANDIDATE_NUM=2

IMAGE_FOLDER=/mnt/wangchenglong/ganyang/data/data_for_mix_vis/mix_preference/coco2017_flickr30k_comb
DATA="llava_reward"

DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"

OUTPUT=models/reward_test

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT

cp $0 $OUTPUT

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps

deepspeed --include localhost:0,1,2,3,4,5 --master_port 12347 training/reward_model_training/rm_training_main_llava_more.py \
    --max_seq_len 2048 --image_folder ${IMAGE_FOLDER} --template ${TEMPLATE} \
    --data_path ${DATA_PATH} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 8 \
    --lm_reward_model_name_or_path ${LLM} \
    --vision_reward_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 1  --zero_stage $ZERO_STAGE --learning_rate $lr --num_warmup_steps 0.1 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 8 --deepspeed --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} --enable_mmca_attention \
    --lang_decoder_update \
    --model_architecture llava_more \
    --precision bf16 --ranked_candidate_num $CANDIDATE_NUM