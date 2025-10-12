import contextlib
import io
from collections import OrderedDict

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForSequenceClassification

# -------------------------------
# Model Loading
# -------------------------------


def load_text_classifier_with_lora(
    model_name: str,
    num_labels: int,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: list = None,
):
    """
    Load a text classification model with LoRA adapters for parameter-efficient fine-tuning.

    Args:
        model_name: HuggingFace model identifier (e.g., 'distilbert-base-uncased')
        num_labels: Number of classification labels
        lora_rank: LoRA rank (r), controls adapter capacity
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: List of module names to apply LoRA to (default: query and value projections)

    Returns:
        Model with LoRA adapters applied
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_lin", "v_lin"]  # DistilBERT uses q_lin/v_lin

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, lora_config)

    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        model.print_trainable_parameters()

    return model


def load_text_classifier_without_lora(
    model_name: str,
    num_labels: int,
):
    """
    Load a text classification model without LoRA (full fine-tuning).

    Args:
        model_name: HuggingFace model identifier (e.g., 'distilbert-base-uncased')
        num_labels: Number of classification labels

    Returns:
        Text classification model
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

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
