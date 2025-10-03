import io
import torch
import contextlib
from collections import OrderedDict
from transformers import AutoModelForMaskedLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

# -------------------------------
# Model Loading
# -------------------------------


def load_model_with_lora(
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list = ["query", "value"],
):
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        model.print_trainable_parameters()

    return model


def load_model(
    model_name: str,
):
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    return model


# -------------------------------
# State Dict (LoRA + Full)
# -------------------------------


def get_lora_state_dict(model):
    """
    Return LoRA adapter weights as a list of numpy arrays.
    """
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for val in state_dict.values()]


def set_lora_state_dict(model, state_list):
    """
    Load LoRA adapter weights from a list of numpy arrays.
    """
    keys = get_peft_model_state_dict(model).keys()
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, state_list)})
    set_peft_model_state_dict(model, state_dict)
