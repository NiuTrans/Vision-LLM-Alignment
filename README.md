# Vision-LLM Alignemnt Training (SFT+PPO/DPO)
Vision-LLM-Alignment is a project designed to implement alignment training for visual large language models (LLMs).
This includes SFT training, reward model training, and PPO/DPO training.
If additional alignment algorithms need to be supported, please raise them in an issue.


## Installation
You can use anaconda/miniconda to install packages needed for this project.
```bash
pip install -r requirements.txt
```

## Preparing Models and Datasets
### Models
Vision-LLM requires both a vision encoder and a language model.
Its architecture is depicted in the [figure](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-VisualChat/assets/model.png).

### Datasets
We have tentatively implemented all alignment training based on this LLaVA dataset format. 
Some samples can be found in the [data folder](https://github.com/wangclnlp/Vision-LLM-Alignment/tree/master/data).

## Training Models
### Supervised Fine-tuning (SFT)
```Shell
# Please refer to the run_sft.sh script for an example.

deepspeed --include localhost:${DEVICE} --master_port 12345 training/sft_training/sft_main.py \
    --max_seq_len ${SEQ_LEN} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --template ${TEMPLATE} \
    --dataset_names ${DATA} \
    --dataset_samples ${DATA_SAMPLE} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --data_train_split_ratio ${DATA_TRAIN_SPLIT_RATIO} \
    --max_num_image_per_sample ${BATCH_SIZE} \
    --eval_step ${EVAL_STEP} \
    --lm_model_name_or_path  ${LLM} \
    --vision_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing \
    --vis_proj [baseline or vit or ...] \
    --gradient_accumulation_steps ${GRAD_STEP} \
    --zero_stage $ZERO_STAGE \
    --learning_rate $lr \
    --num_warmup_steps 0.1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --deepspeed --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} \
    --enable_mmca_attention \
    --lang_decoder_update \
    --precision bf16 
```

### Reward Model Training
```Shell
# Please refer to the run_rm_training.sh script for an example.

deepspeed --include localhost:$DEVICE --master_port 12345 training/reward_model_training/rm_training_main.py \
     --max_seq_len ${SEQ_LEN} \
    --data_path ${DATA_PATH} \
    --eval_data_path ${EVAL_DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --dataset_names ${DATA} \
    --dataset_samples ${DATA_SAMPLE} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --max_num_image_per_sample 8 \
    --lm_reward_model_name_or_path ${LLM} \
    --vision_reward_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing --vis_proj baseline \
    --gradient_accumulation_steps 1 \
    --zero_stage $ZERO_STAGE \
    --learning_rate $lr \
    --num_warmup_steps 0.1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --deepspeed \
    --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} \
    --enable_mmca_attention \
    --lang_decoder_update \
    --precision bf16 \
    --ranked_candidate_num $candidate_num
```
### Direct Pereference Optimization (DPO)
```Shell
# Please refer to the run_dpo_training.sh script for an example.

deepspeed --include localhost:$DEVICE --master_port 12345 training/dpo_training/dpo_training_main.py \
    --max_seq_len ${SEQ_LEN} \
    --data_path ${DATA_PATH} \
    --dataset_names ${DATA} \
    --dataset_samples ${DATA_SAMPLE} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --max_num_image_per_sample 8 \
    --lm_model_name_or_path  ${LLM} \
    --vision_model_name_or_path ${VISION_MODEL} \
    --from_checnkpoint ${FROM_CHECKPOINT} \
    --gradient_checkpointing \
    --vis_proj [baseline or vit or ...] \
    --gradient_accumulation_steps 8  \
    --zero_stage $ZERO_STAGE \
    --learning_rate $lr \
    --num_warmup_steps 0.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --deepspeed \
    --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} \
    --enable_mmca_attention \
    --lang_decoder_update \
    --precision bf16 \
    --ranked_candidate_num $candidate_num  
```
### Reinforcement Learning from Human Feedback (RLHF)
```Shell
# Please refer to the run_ppo_training.sh script for an example.

deepspeed --include localhost:$DEVICE --master_port 12345 training/ppo_training/ppo_main.py \
    --max_seq_len ${SEQ_LEN} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --dataset_names ${DATA} \
    --dataset_samples ${DATA_SAMPLE} \
    --data_train_split_ratio ${TRAIN_SPLIT_RATIO} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --max_num_image_per_sample 8 \
    --template ${TEMPLATE} \
    --lm_reward_model_name_or_path  ${LLM} \
    --vision_reward_model_name_or_path ${VISION_MODEL} \
    --gradient_checkpointing \
    --vis_proj baseline \
    --gradient_accumulation_steps 4 \
    --num_warmup_steps 0.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --deepspeed \
    --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} \
    --ppo_epochs 4 \
    --enable_mmca_attention \
    --lang_decoder_update \
    --precision bf16 \
    --sft_model_ckpt_path $sft_model_ckpt_path \
    --reward_model_ckpt_path $reward_model_ckpt_path \
    --lm_model_name_or_path $LLM \
    --vision_model_name_or_path $VISION_MODEL \
    --lm_reward_model_name_or_path $lm_reward_model_name_or_path \
    --vision_reward_model_name_or_path $vision_reward_model_name_or_path \
    --actor_zero_stage $actor_zero_stage \
    --critic_zero_stage $critic_zero_stage \
    --image_folder ${IMAGE_FOLDER} \
    --actor_learning_rate $ACTOR_LEARNING_RATE \
    --critic_learning_rate $CRITIC_LEARNING_RATE 
```
### Evaluation
```Shell
# Please refer to the run_predict.sh script for an example.

deepspeed --include localhost:$DEVICE --master_port 12345 ./eval/predict.py \
    --max_seq_len ${MAX_LENGTH} \
    --data_path ${DATA_PATH} \
    --dataset_names ${DATA} \
    --dataset_samples ${DATA_SAMPLE} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} \
    --precision bf16 \
    --enable_mmca_attention \
    --sft_model_ckpt_path ${CKPT_PATH} \
    --template ${TEMPLATE} \
    --lm_model_name_or_path ${LLM} \
    --vision_model_name_or_path ${VISION_MODEL} \
    --batch_size ${BATCH_SIZE} \
    --image_folder ${IMAGE_FOLDER} \
    --output_path ${OUTPUT}
```

## Supported Models
| LLM | Model size |
|:---:|:---:|
| LLaMA-2 | 7B/13B/70B |
| LLaMA-3 | 8B/70B |

Note: Other LLMs with the same architecture as LLaMA-2/3 are also supported.

| Vision Model |
|:---:|
| clip-vit-large-patch14 |
| clip-vit-large-patch14-336 |

## Supported Traing Modes

| Method | Full | LoRA |
|:---:|:---:|:---:|
| SFT |  √  | √ |
| RM  |  √  | √ |
| DPO |  √  | √ |
| PPO |  √  |  |

## Acknowledgement
We commence by utilizing the exceptional codebase provided by [DeepSpeed-VisualChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat) 🌹🌹🌹.

We thank the following papers:
```bash
[1] Ouyang, Long, et al. "Training language models to follow instructions with human feedback." Advances in neural information processing systems 35 (2022): 27730-27744.
[2] Rafailov, Rafael, et al. "Direct preference optimization: Your language model is secretly a reward model." Advances in Neural Information Processing Systems 36 (2024).
[3] Liu, Haotian, et al. "Visual instruction tuning." Advances in neural information processing systems 36 (2024).
```


