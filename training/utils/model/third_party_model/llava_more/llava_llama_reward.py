from typing import List, Optional, Tuple, Union


import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import sys
import os

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLMWithValueHead(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, tokenizer):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.v_head =  nn.Linear(config.hidden_size, 1, bias=False)
        self.tokenizer = tokenizer
        self.PAD_ID = tokenizer.pad_token_id
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None # warning: this is required for inference and last versions of transformers
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # if self.cache_position: self.cache_position = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
        rewards = self.v_head(outputs.hidden_states[-1])
        bs = rewards.size(0)
        seq_len = labels.shape[1]

        end_reward_scores = []
        for i in range(bs):
            sub_input_ids = labels[i]
            value = rewards[i]
            c_inds = (sub_input_ids != self.PAD_ID).nonzero()
            c_ind = c_inds[-1].item() if len(c_inds)>0 else seq_len
            end_reward_scores.append(value[c_ind])
        return end_reward_scores


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLMWithValueHead)
