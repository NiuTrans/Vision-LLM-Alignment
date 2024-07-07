# Vision-LLM Alignemnt Training (SFT+PPO/DPO)
Vision-LLM-Alignment is a project designed to implement alignment training for visual large language models (LLMs).
This includes SFT training, reward model training, and PPO/DPO training.
If additional alignment algorithms need to be supported, please raise them in an issue.

## Changelog
- [2024/07/07] We support the direct loading of a LLaVA model during the SFT training phase. You just need to set the `model_architecture` parameter to "llava" and specify the LLaVA model path with `from_checkpoint`. Support for this functionality during the DPO, RM training, and PPO junction phases will be introduced soon.

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
bash run_sft.sh 
```

### Reward Model Training
```Shell
bash run_rm_training.sh
```
### Direct Pereference Optimization (DPO)
```Shell
bash run_dpo_training.sh
```
### Reinforcement Learning from Human Feedback (RLHF)
```Shell
bash run_ppo_training.sh
```
### Evaluation
```Shell
bash run_predict.sh 
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


