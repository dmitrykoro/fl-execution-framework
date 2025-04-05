import io
import torch
import contextlib
from torch.optim import AdamW
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
        output = buf.getvalue()

    return model

def load_model(
    model_name: str,
):
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    return model

# -------------------------------
# Training and Evaluation
# -------------------------------

def train(model, trainloader, epochs, device, server_round=1, learning_rate=5e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0

        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if hasattr(outputs, "logits"):
                preds = torch.argmax(outputs.logits, dim=-1)
                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        avg_loss = total_loss / len(trainloader)
        accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy

def test(model, testloader, device, server_round=1):
    model.to(device)
    model.eval()

    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            outputs = model(**batch)
            loss = outputs.loss.item()
            total_loss += loss

            if hasattr(outputs, "logits"):
                preds = torch.argmax(outputs.logits, dim=-1)
                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy

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

def get_state_dict(model):
    """
    Return full model state dict (non-LoRA) as list of numpy arrays.
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_state_dict(model, state_list):
    """
    Load full model weights from list of numpy arrays.
    """
    keys = model.state_dict().keys()
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, state_list)})
    model.load_state_dict(state_dict, strict=True)
