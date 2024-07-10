from .third_party_model.hf_model.modeling_llava import LlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from .modeling_dsvl import create_dsvl_model_and_transforms
from ..data import DST
# You can design (or specify) the architecture of vision LLM.
def build_model(text_tokenizer=None,
                ds_config=None,
                args=None):
    
    if args.model_architecture=='default':
        model, image_processor, tokenizer = create_dsvl_model_and_transforms(
            text_tokenizer=text_tokenizer,
            args=args,
            ds_config=ds_config)
        return model, image_processor, tokenizer
    
    elif args.model_architecture=="llava":
        model = LlavaForConditionalGeneration.from_pretrained(
                args.from_checkpoint, 
                low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(args.from_checkpoint)
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
        
        tokenizer.add_tokens(DST.special_token_list, special_tokens=True)
        model.language_model.config.vocab_size = len(tokenizer)
        model.language_model.resize_token_embeddings(len(tokenizer))

        return model, image_processor, tokenizer
    
    else:
        assert "Please add this model architeacture in build_model.py!"
