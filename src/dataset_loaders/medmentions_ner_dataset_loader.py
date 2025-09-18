import os, json
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
import torch

class _JsonlNERDataset(Dataset):
    def __init__(self, path): self.rows=[json.loads(x) for x in open(path,"r",encoding="utf-8")]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

class _GPT2NERCollator:
    def __init__(self, label2id, max_length=512):
        self.label2id = label2id
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.max_length = max_length
    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        enc = self.tok(texts, return_offsets_mapping=True, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        O = self.label2id["O"]
        labels = torch.full_like(enc["input_ids"], O)
        for i,b in enumerate(batch):
            tags = b["labels_str"]
            L = enc["input_ids"].shape[1]
            if len(tags) >= L:
                tag_ids = [self.label2id.get(t,O) for t in tags[:L]]
            else:
                tag_ids = [self.label2id.get(t,O) for t in tags] + [O]*(L-len(tags))
            labels[i,:] = torch.tensor(tag_ids[:L])
        enc.pop("offset_mapping")
        enc["labels"] = labels
        return enc

class MedMentionsNERDatasetLoader:
    def __init__(self, dataset_dir: str, num_of_clients: int, batch_size: int=8, max_length: int=512):
        self.dataset_dir = dataset_dir
        self.num_clients = num_of_clients
        self.batch_size = batch_size
        with open(os.path.join(dataset_dir, "label2id.json"), "r") as f: self.label2id = json.load(f)
        self.collate = _GPT2NERCollator(self.label2id, max_length=max_length)

    def _client_paths(self, cid):
        cdir = os.path.join(self.dataset_dir, f"client_{cid}")
        return (os.path.join(cdir,"train.jsonl"), os.path.join(cdir,"val.jsonl"))

    def load_datasets(self):
        trainloaders, valloaders = [], []
        for cid in range(self.num_clients):
            tr, va = self._client_paths(cid)
            train = _JsonlNERDataset(tr)
            val   = _JsonlNERDataset(va)
            trainloaders.append(DataLoader(train, batch_size=self.batch_size, shuffle=True,  collate_fn=self.collate))
            valloaders.append(  DataLoader(val,   batch_size=self.batch_size, shuffle=False, collate_fn=self.collate))
        return trainloaders, valloaders
