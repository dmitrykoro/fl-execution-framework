import torch
from collections import OrderedDict
from transformers import GPT2Config, GPT2TokenizerFast, GPT2ForTokenClassification
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

def load_model(model_name: str = "gpt2", num_labels: int = 2, use_lora: bool = True,
               lora_r: int=8, lora_alpha: int=16, lora_dropout: float=0.05):
    cfg = GPT2Config.from_pretrained(model_name, num_labels=num_labels, pad_token_id=50256)
    model = GPT2ForTokenClassification.from_pretrained(model_name, config=cfg)
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    if use_lora:
        target_modules = ["c_attn", "c_proj"]
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          bias="none", task_type="TOKEN_CLS", target_modules=target_modules)
        model = get_peft_model(model, lcfg)
    return model

# LoRA helpers (work for any PEFT model)
def get_peft_model_state(model):
    return get_peft_model_state_dict(model)

def set_peft_model_state(model, state_list):
    keys = get_peft_model_state_dict(model).keys()
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, state_list)})
    set_peft_model_state_dict(model, state_dict)
