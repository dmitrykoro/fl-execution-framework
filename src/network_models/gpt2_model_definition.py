import torch

from transformers import AutoModelForTokenClassification, AutoConfig
from peft import LoraConfig, get_peft_model
from transformers import GPT2ForTokenClassification


def load_gpt2_ner_model(num_labels: int, id2label: dict, label2id: dict):
    cfg = AutoConfig.from_pretrained("gpt2", num_labels=num_labels, id2label=id2label, label2id=label2id)
    model = GPT2ForTokenClassification.from_pretrained("gpt2", num_labels=num_labels)
    model.config.pad_token_id = cfg.eos_token_id if cfg.pad_token_id is None else cfg.pad_token_id
    return model


def load_gpt2_ner_with_lora(
        num_labels: int, id2label: dict, label2id: dict,
        r: int = 16, alpha: int = 32, dropout: float = 0.1
):
    base = load_gpt2_ner_model(num_labels, id2label, label2id)
    peft_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        task_type="TOKEN_CLS",
        target_modules=["c_attn", "c_proj"],
        inference_mode=False,
    )
    return get_peft_model(base, peft_cfg)
