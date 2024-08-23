from .third_party_model.hf_model.modeling_llava import LlavaForConditionalGeneration
from .third_party_model.hf_model.configuration_llava import LlavaConfig
from .third_party_model.hf_model.modeling_llava_next import LlavaNextForConditionalGeneration
# from transformers import LlavaNextForConditionalGeneration
from .third_party_model.hf_model.configuration_llava_next import LlavaNextConfig
from transformers import AutoTokenizer, AutoProcessor
from .modeling_dsvl import create_dsvl_model_and_transforms
from ..data import DST
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
