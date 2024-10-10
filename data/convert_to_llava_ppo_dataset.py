import json
import random

# convert multi-turn dialogue to single dialogue format.

template = "llama_3" # llama_2, vicuna, defalut

source_data = open('data/llava/llava_instruct_150k.json', 'r', encoding='utf-8')
source_data = json.load(source_data)

defalut_roles = ["\n\n### Question:\n", "\n\n### Answer:\n",""]
llama2_roles = ["[INST] "," [/INST] "," </s><s>"]
llama3_roles = ["<|start_header_id|>user<|end_header_id|>\n\n","<|start_header_id|>assistant<|end_header_id|>\n\n","<|eot_id|>"]
vicuna_roles = ["USER: ","ASSISTANT: ","</s>"]
mistral_roles = ["[INST]","[/INST]"," </s>"]
llama3_2_roles = ["<|start_header_id|>user<|end_header_id|>\n\n","<|start_header_id|>assistant<|end_header_id|>\n\n","<|eot_id|>"]

if template == "defalut":
    roles = defalut_roles
elif template == "llama_2":
    roles = llama2_roles
elif template == "llama_3":
    roles = llama3_roles
elif template == "vicuna":
    roles = vicuna_roles
elif template == "mistral":
    roles = mistral_roles
elif template == "llama_3.2":
    roles = llama3_2_roles

all_data = []

for line in source_data:
    sample_id = line["image"].replace(".jpg", "")
    image = line["image"]

    human_string = ""

    if len(line["conversations"]) > 2:
        for index, conv in enumerate(line["conversations"][:-1]):
            token_of_end = ""
            if index%2 == 0 and (template == "llama_3" or template == "llama_3.2") and index != (len(line["conversations"])-2):
                token_of_end = roles[2]
            if index%2 == 1:
                token_of_end = roles[2]
            if index == 0:
                human_string += conv["value"].strip("\n")
            else:
                human_string += roles[index%2]
                human_string += conv["value"].strip("\n")
            human_string += token_of_end
        gpt_string = line["conversations"][-1]["value"]
    else:
        human_string += line["conversations"][0]["value"]
        gpt_string = line["conversations"][1]["value"]


    all_data.append(
        {
            "id": sample_id,
            "image": image,
            "conversations": [
                {
                    "from": "human",
                    "value": human_string
                },
                {
                    "from": "gpt",
                    "value": gpt_string
                }
            ]
        }
    )

with open('data/llava/ppo/llava_instruct_150k_for_ppo_training.json', 'w', encoding='utf-8') as file:
    json_str = json.dump(all_data, file, ensure_ascii=False, indent=4)


    
                