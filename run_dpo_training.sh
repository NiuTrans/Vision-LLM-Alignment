#!/bin/bash

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=none
LLM=none

FROM_CHECKPOINT=your-llava-path

TEMPLATE=llava

CANDIDATE_NUM=2
MODEL_ARCHITECTURE="llava"
EPOCH=1
ZERO_STAGE=2
lr=1e-6

DATA="llava_reward"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"


DATA_PATHs=(
data/reward_samples.json
)
IMAGE_FOLDERs=(
data/reward_samples_folders
)

OUTPUTs=(
models/dpo_test
)

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

array_num=${#OUTPUTs[@]}

for ((i=0; i<$array_num; i++))
do

DATA_PATH=${DATA_PATHs[i]}
IMAGE_FOLDER=${IMAGE_FOLDERs[i]}
OUTPUT=${OUTPUTs[i]}

mkdir -p $OUTPUT

cp $0 $OUTPUT

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12320 training/dpo_training/dpo_training_main.py --max_seq_len 256 \
    --data_path ${DATA_PATH} --image_folder ${IMAGE_FOLDER} --template ${TEMPLATE} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --max_num_image_per_sample 8 \
    --lm_model_name_or_path  ${LLM} \
    --vision_model_name_or_path ${VISION_MODEL} \
    --model_architecture ${MODEL_ARCHITECTURE} \
    --from_checkpoint ${FROM_CHECKPOINT} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 2  --zero_stage $ZERO_STAGE --learning_rate $lr --num_warmup_steps 0.1 \
    --per_device_train_batch_size 4  --deepspeed --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} --enable_mmca_attention --lang_decoder_update \
    --precision bf16 --ranked_candidate_num $CANDIDATE_NUM > $OUTPUT/training.log &

done