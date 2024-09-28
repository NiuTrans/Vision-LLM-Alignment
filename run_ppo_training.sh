#!/bin/bash
CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=None
LLM=None

sft_model_ckpt_path=your-llava-path

TEMPLATE=llava
MODEL_ARCHITECTURE=llava

lm_reward_model_name_or_path=$LLM
vision_reward_model_name_or_path=$VISION_MODEL


actor_zero_stage=2
critic_zero_stage=3

ACTOR_LEARNING_RATE=1e-6
CRITIC_LEARNING_RATE=2e-5

MAX_GENERATION_LANGTH_OF_SAMPLING=512

EPOCH=1

IMAGE_FOLDER=data/ppo_samples_image_folder
DATA_PATH=data/ppo_samples.json

TRAIN_SPLIT_RATIO=0.999

DATA="llava_ppo"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"

reward_model_ckpt_paths=(
your-reward-model-path
)
OUTPUTs=(
models/ppo-test
)

array_num=${#reward_model_ckpt_paths[@]}

for ((i=0; i<$array_num; i++))
do

OUTPUT=${OUTPUTs[i]}
reward_model_ckpt_path=${reward_model_ckpt_paths[i]}

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

mkdir -p $OUTPUT

cp $0 $OUTPUT

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12346 training/ppo_training/ppo_main.py --max_seq_len 2048 \
    --data_path ${DATA_PATH} --image_folder ${IMAGE_FOLDER} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --data_train_split_ratio ${TRAIN_SPLIT_RATIO} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 1 \
    --template ${TEMPLATE} \
    --lm_reward_model_name_or_path  ${LLM} \
    --vision_reward_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 2 --num_warmup_steps 0.1 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --save_step 500 --eval_step 9999 \
    --max_training_step 500 \
    --skip_actor_model 30 \
    --deepspeed --output_dir $OUTPUT  \
    --model_architecture $MODEL_ARCHITECTURE \
    --num_train_epochs ${EPOCH} --ppo_epochs 2 --enable_mmca_attention \
    --lang_decoder_update --precision bf16 \
    --from_checkpoint $sft_model_ckpt_path \
    --reward_base_model $sft_model_ckpt_path \
    --reward_model_ckpt_path $reward_model_ckpt_path \
    --lm_model_name_or_path $LLM \
    --vision_model_name_or_path $VISION_MODEL \
    --lm_reward_model_name_or_path $lm_reward_model_name_or_path \
    --vision_reward_model_name_or_path $vision_reward_model_name_or_path \
    --actor_zero_stage $actor_zero_stage --critic_zero_stage $critic_zero_stage \
    --actor_learning_rate $ACTOR_LEARNING_RATE --critic_learning_rate $CRITIC_LEARNING_RATE \
    --max_generation_length_of_sampling ${MAX_GENERATION_LANGTH_OF_SAMPLING} > $OUTPUT/training.log &

done