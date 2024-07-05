#!/bin/bash

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=base_models/vision_encoder/clip-vit-large-patch14
LLM=base_models/llama-3-8b-Instruct

FROM_CHECKPOINT=models/sft_test/epoch-3

TEMPLATE=llama_3

IMAGE_FOLDER=data/coco_2017/

EPOCH=3
ZERO_STAGE=2
lr=1e-5

DATA_PATH=data/reward_samples.json
CANDIDATE_NUM=2

DATA="llava_reward"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"

OUTPUT=models/dpo_test

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
# Note: when training with multiple candidate outputs, you should employ the training/dpo_training/multi_candidate_dpo_training_main.py

deepspeed --include localhost:0 --master_port 12348 training/dpo_training/dpo_training_main.py --max_seq_len 2048 \
    --data_path ${DATA_PATH} --image_folder ${IMAGE_FOLDER} --template ${TEMPLATE} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 8 \
    --lm_model_name_or_path  ${LLM} \
    --vision_model_name_or_path ${VISION_MODEL} \
    --from_checkpoint ${FROM_CHECKPOINT} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 8  --zero_stage $ZERO_STAGE --learning_rate $lr --num_warmup_steps 0.1 \
    --per_device_train_batch_size 2  --deepspeed --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} --enable_mmca_attention --lang_decoder_update \
    --precision bf16 --ranked_candidate_num $CANDIDATE_NUM 