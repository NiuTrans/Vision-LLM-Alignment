# Evaluate on Custom Datasets

## Data Preparation

The data under evaluation should be in a JSON list. And each element of data should be a JSON dictionary including "id", "image", "conversations" at least. For instance, the data should in the format below:

```JSON
[
    {
        "id": ... ,
        "image": [name of the image file] or [null],
        "conversations": [
            {
                "from": "human",
                "value": [instruction or question]
            },
            {
                "from": "gpt",
                "value": [answer or label]
            }
        ]
    },
    ... 
]
```

## Run with Shell

Run with the shell script shown below:

```Shell
#!/bin/bash

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

deepspeed --include localhost:[DEVICE] --master_port [PORT]./eval/predict.py \
    --max_seq_len [MAX_LENGTH] \
    --data_path [DATA_PATH] \
    --dataset_names [DATA] --dataset_samples [DATA_SAMPLE] --dataset_concatenate_samples [IMAGE_PER_SAMPLE] \
    --precision [fp16 or bf16] --enable_mmca_attention \
    --sft_model_ckpt_path [CKPT_PATH] \
    --template [TEMPLATE] \
    --lm_model_name_or_path [LLM] \
    --vision_model_name_or_path [VISION_MODEL] \
    --batch_size [BATCH_SIZE] \
    --image_folder [IMAGE_FOLDER] \
    --output_path [OUTPUT]
    ...
```