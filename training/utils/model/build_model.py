from .third_party_model.hf_model.modeling_llava import LlavaForConditionalGeneration
from .third_party_model.hf_model.configuration_llava import LlavaConfig
from .third_party_model.hf_model.modeling_llava_next import LlavaNextForConditionalGeneration
# from transformers import LlavaNextForConditionalGeneration
from .third_party_model.hf_model.configuration_llava_next import LlavaNextConfig
from transformers import AutoTokenizer, AutoProcessor
from .modeling_dsvl import create_dsvl_model_and_transforms
from ..data import DST

import torch
from .third_party_model.llava_more.llava_llama import LlavaLlamaForCausalLM
from .third_party_model.llava_more.llava_llama_reward import LlavaLlamaForCausalLMWithValueHead
from .third_party_model.llava_more.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN

# You can design (or specify) the architecture of vision LLM.
def build_model(text_tokenizer=None,
                ds_config=None,
                args=None,
                model_architecture=None,
                from_checkpoint=None):
    
    if model_architecture is None:
        model_architecture = args.model_architecture

    if from_checkpoint is None:
        from_checkpoint = args.from_checkpoint

    if model_architecture=='default' or model_architecture == "debug_wo_model":
        model, image_processor, tokenizer = create_dsvl_model_and_transforms(
            text_tokenizer=text_tokenizer,
            args=args,
            ds_config=ds_config)
        return model, image_processor, tokenizer
    
    elif model_architecture=="llava":
        model = LlavaForConditionalGeneration.from_pretrained(
                from_checkpoint, 
                low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(from_checkpoint)

        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'left'
        
        # freeze parameters
        model.vision_tower.requires_grad_(False)
        model.multi_modal_projector.requires_grad_(True)

        if args.lang_decoder_update:
            model.language_model.requires_grad_(True)
        else:
            model.language_model.requires_grad_(False)

        return model, image_processor, tokenizer
    elif model_architecture=="llava_next":
        model = LlavaNextForConditionalGeneration.from_pretrained(
                from_checkpoint, 
                low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(from_checkpoint)

        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'left'
        
        # freeze parameters
        model.vision_tower.requires_grad_(False)
        model.multi_modal_projector.requires_grad_(True)

        if args.lang_decoder_update:
            model.language_model.requires_grad_(True)
        else:
            model.language_model.requires_grad_(False)

        return model, image_processor, tokenizer
    else:
        assert "Please add this model architeacture in build_model.py!"


def build_model_llava_more(
        model_path,
        model_name,
        # device_map="auto",
        # device="cuda",
        mlp_path=None,
        torch_dtype=torch.float16
):
    # kwargs = {
    #     "device_map":device_map,
    #     "torch_dtype":torch_dtype     
    # }

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False
    )
    model = LlavaLlamaForCausalLMWithValueHead.from_pretrained(
        model_path,
        tokenizer=tokenizer,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16
    )

    image_processor = None
    if 'llava' in model_name.lower() or mlp_path is not None:
        mm_use_im_start_end = getattr(
            model.config,
            "mm_use_im_start_end",
            False
        )
        mm_use_im_patch_token = getattr(
            model.config,
            "mm_use_im_patch_token",
            True
        )
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [
                    DEFAULT_IM_START_TOKEN,
                    DEFAULT_IM_END_TOKEN
                ],
                special_tokens=True
            )
        if mm_use_im_patch_token:
            tokenizer.add_tokens(
                [
                   DEFAULT_IMAGE_PATCH_TOKEN
                ],
                special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()

        if not vision_tower.is_loaded:
            vision_tower.load_model()
        # if device_map != 'auto':
        #     vision_tower.to(device=device_map, dtype=torch.float16)
        vision_tower.to(torch.bfloat16)
        image_processor = vision_tower.image_processor
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
            
        if mlp_path is not None:
            print('Loading mm projector weights...')
            mm_projector_weights = torch.load(mlp_path)
            new_dict= {}
            new_keys= ['0.weight', '0.bias', '2.weight', '2.bias']
            for el, key in enumerate(new_keys):
                new_dict[key] = mm_projector_weights[list(mm_projector_weights.keys())[el]]

            model.model.mm_projector.load_state_dict(new_dict)
            # model.model.mm_projector.to(device=device_map, dtype=torch.float16)
        model.model.mm_projector.to(torch.bfloat16)
        return tokenizer, model, image_processor, context_len
