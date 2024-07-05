#!/bin/bash

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=base_models/vision_encoder/clip-vit-large-patch14
LLM=base_models/llama-3-8b-Instruct

TEMPLATE=llama_3

DATA_PATH=data/sft_samples.json
IMAGE_FOLDER=data/coco_2017/
DATA_TRAIN_SPLIT_RATIO=0.9

OUTPUT=models/sft_test

EPOCH=3
ZERO_STAGE=2
lr=2e-3

DATA="llava_sft"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT

cp $0 $OUTPUT

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
# Note: we keep the visual encoder weights and LLM decoder weights frozen in the process of pretraining.

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12346 training/sft_training/sft_main.py --max_seq_len 2048 \
    --data_path ${DATA_PATH} --image_folder ${IMAGE_FOLDER} --template ${TEMPLATE}\
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --data_train_split_ratio ${DATA_TRAIN_SPLIT_RATIO} --max_num_image_per_sample 8 --eval_step 500 \
    --lm_model_name_or_path  ${LLM} \
    --vision_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 1  --zero_stage $ZERO_STAGE --learning_rate $lr --num_warmup_steps 0.1 \
    --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --deepspeed --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} --enable_mmca_attention \
    --precision bf16 