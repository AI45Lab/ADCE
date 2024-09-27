import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig




def load_models_config(config_path='../model_data_config.json'):
    with open(config_path, 'r') as file:
        data = json.load(file)
    return data['models']


def load_model(model_name, gpu,quantization=False):
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    models_config = load_models_config()
    model_path = models_config.get(model_name, None)
    if model_path is None:
        raise ValueError(f"Model {model_name} not found in configuration.")
    
    if quantization:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', output_attentions=True,torch_dtype=torch.float16, quantization_config=nf4_config,attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', output_attentions=True,torch_dtype=torch.float16,attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    mask_token = '<mask>'
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'mask_token': mask_token})
        model.resize_token_embeddings(len(tokenizer))
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        model.config.mask_token_id = mask_token_id
    print(f"Model {model_name} and Tokenizer loaded successfully on {device}.")
    return model, tokenizer

