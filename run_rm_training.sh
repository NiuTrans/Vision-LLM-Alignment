#!/bin/bash
# This is an example where we use the lava model to train a reward model.

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=none
LLM=none

FROM_CHECKPOINT=/localnvme/application/sc_new/wangchenglong_56/base_models/llama-3.2-11b-vision-instruct

MODEL_ARCHITECTURE="llama-3.2-vision"

TEMPLATE="llama-3.2-vision"

EPOCH=1
ZERO_STAGE=3

lr=1e-6

# if you do train a reward based on a pre-trained reward model, 
# this parameter does not need to be set
TRAINED_REWARD_MODEL=none 

OUTPUT=models/test

DATA_PATH=/localnvme/application/sc_new/wangchenglong_56/rlhf_llama_vision/data/RLAIF-V-Dataset/rlaif_v_dataset_train.json
EVAL_DATA_PATH=/localnvme/application/sc_new/wangchenglong_56/rlhf_llama_vision/data/RLAIF-V-Dataset/rlaif_v_dataset_test.json

IMAGE_FOLDER=/localnvme/application/sc_new/wangchenglong_56/rlhf_llama_vision/data/RLAIF-V-Dataset/images

CANDIDATE_NUM=2

DATA="llava_reward"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT

cp $0 $OUTPUT

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps

nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12335 training/reward_model_training/rm_training_main.py \
    --max_seq_len 2048 --image_folder ${IMAGE_FOLDER} --template ${TEMPLATE} \
    --data_path ${DATA_PATH} --eval_data_path ${EVAL_DATA_PATH} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 8 \
    --lm_reward_model_name_or_path ${LLM} \
    --vision_reward_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 2  --zero_stage $ZERO_STAGE --learning_rate $lr --num_warmup_steps 0.1 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 8 --eval_step 200 \
    --deepspeed --output_dir $OUTPUT --num_train_epochs ${EPOCH} \
    --lang_decoder_update --enable_mmca_attention --model_architecture ${MODEL_ARCHITECTURE} \
    --trained_reward_model $TRAINED_REWARD_MODEL --save_step 9900 \
    --precision bf16 --ranked_candidate_num $CANDIDATE_NUM --from_checkpoint ${FROM_CHECKPOINT} > $OUTPUT/training.log &