#!/bin/bash
# This is an example where we use the lava model to condut the prediction.

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

VISION_MODEL=none
LLM=none

CKPT_PATHs=(
your-llava-path
)

TEMPLATE=llava
MODEL_ARCHITECTURE=llava
 
# sciQA
DATA_PATH=data/prediction_sample/sci_qa_test.txt
IMAGE_FOLDER=data/prediction_sample/sci_qa_test_image
OUTPUT_TAG=/res/sciqa_with_image.res

DATA="llava_predict"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"
BATCH_SIZE=1

TOPK=50
TOPP=0.95
MAX_NEW_TOKENS=1024
NUM_RETURN_SEQUENCES=1
TEMPERATURE=0.0

DEVICES=(7)

ARRAY_LENGTH=${#CKPT_PATHs[@]}

for (( i=0; i<${ARRAY_LENGTH}; i++ )); do

CKPT_PATH=${CKPT_PATHs[i]}

mkdir $CKPT_PATH/res

OUTPUT=$CKPT_PATH/$OUTPUT_TAG
rm ${OUTPUT}

DEVICE=${DEVICES[i]}

# we assume the batch size is 128, which means Num_GPU * per_device_train_batch_size * gradient_accumulation_steps
nohup deepspeed --include localhost:$DEVICE --master_port $((12345 + DEVICE)) ./eval/predict.py \
    --max_seq_len 2048 \
    --data_path ${DATA_PATH} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --precision bf16 --enable_mmca_attention \
    --from_checkpoint ${CKPT_PATH} \
    --template ${TEMPLATE} \
    --model_architecture ${MODEL_ARCHITECTURE} \
    --lm_model_name_or_path $LLM \
    --num_return_sequences 1 \
    --vision_model_name_or_path $VISION_MODEL \
    --batch_size ${BATCH_SIZE} \
    --image_folder ${IMAGE_FOLDER} \
    --output_path ${OUTPUT} \
    --do_sample \
    --topk ${TOPK} \
    --topp ${TOPP} \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --temperature $TEMPERATURE > $CKPT_PATH/generating.log &
done 
