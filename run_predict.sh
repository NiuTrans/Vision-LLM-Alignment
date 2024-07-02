#!/bin/bash

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=base_models/vision_encoder/clip-vit-large-patch14
LLM=base_models/llama-2-7b-hf

TEMPLATE=llama_3

IMAGE_FOLDER=data/coco2017_flickr30k_comb

CKPT_PATH=models/test/epoch-5

TEMPLATE=llama_3

OUTPUT=output/test_tmp.res

DATA_PATH=data/llava/ppo/ppo_test.json

DATA="llava_predict"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"
BATCH_SIZE=1

rm ${OUTPUT}

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps

deepspeed --include localhost:3 --master_port 12362 ./eval/predict.py \
    --max_seq_len 2048 \
    --data_path ${DATA_PATH} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --precision bf16 --enable_mmca_attention \
    --sft_model_ckpt_path ${CKPT_PATH} \
    --template ${TEMPLATE} \
    --lm_model_name_or_path $LLM \
    --vision_model_name_or_path $VISION_MODEL \
    --batch_size ${BATCH_SIZE} \
    --image_folder ${IMAGE_FOLDER} \
    --output_path ${OUTPUT}
