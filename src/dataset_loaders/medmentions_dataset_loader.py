import os, glob, json
from typing import List, Tuple
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, DataCollatorForTokenClassification

class MedMentionsNERDatasetLoader:
    """
    Expects JSON arrays with fields: id, document_id, text (optional), tokens, ner_tags (BIO)
    Returns per-client DataLoaders with GPT-2 tokenization + aligned labels.
    """

    IGNORE_LABEL = -100  # for subword tokens
    MAX_LEN = 512        # change if you plan to window to 1024 later

    def __init__(
        self,
        dataset_dir: str,
        num_of_clients: int,
        batch_size: int,
        training_subset_fraction: float,
        model_name: str = "gpt2",
    ) -> None:
        self.dataset_dir = dataset_dir
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction
        self.model_name = model_name

        # Build a global label list so ALL clients share the same mapping
        self.label_list = self._scan_all_labels()
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        # Tokenizer (GPT-2 has no PAD by default)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2",
            add_prefix_space=True,
            padding_side="right",
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Let the collator pad dynamically
        self.collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            label_pad_token_id=self.IGNORE_LABEL,
        )

    def _scan_all_labels(self) -> List[str]:
        labels = set(["O"])
        for client_folder in sorted(os.listdir(self.dataset_dir), key=lambda s: int(s.split("_")[1])):
            if client_folder.startswith("."):
                continue
            for split in ("train", "validation", "test"):
                for fp in glob.glob(os.path.join(self.dataset_dir, client_folder, f"{split}*.json")):
                    try:
                        data = json.load(open(fp, "r", encoding="utf-8"))
                        for ex in data:
                            labels.update(ex["ner_tags"])
                    except Exception:
                        pass
        return sorted(labels)

    def _encode_align(self, batch):
        """
        Convert word-level tokens/tags to subword-aligned tensors AND
        also carry:
        - doc_id        (str per example)
        - word_length   (# of original word tokens per example)
        """
        enc = self.tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,            # collator pads
            max_length=self.MAX_LEN,
            return_attention_mask=True,
        )

        labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            aligned = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(self.IGNORE_LABEL)          # special/pad tokens
                elif wid != prev_wid:
                    aligned.append(self.label2id[tags[wid]])   # first subword gets tag
                else:
                    aligned.append(self.IGNORE_LABEL)          # ignore non-first subwords
                prev_wid = wid
            labels.append(aligned)

        # NEW: carry these through for strict mention/doc metrics
        doc_ids = [str(d) for d in batch["document_id"]]
        word_lengths = [len(toks) for toks in batch["tokens"]]

        # Return ONLY model fields + the two extras.
        # (We will still remove the original columns in .map(..., remove_columns=orig_cols))
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "doc_id": doc_ids,               # <-- added
            "word_length": word_lengths,     # <-- added
        }


    def _load_one_client(self, client_dir: str) -> Tuple[DataLoader, DataLoader]:
        files_train = glob.glob(os.path.join(client_dir, "train*.json"))
        files_val   = glob.glob(os.path.join(client_dir, "validation*.json"))
        if not files_val:  # fallback if you only have test
            files_val = glob.glob(os.path.join(client_dir, "test*.json"))

        ds = DatasetDict({
            "train": load_dataset("json", data_files=files_train)["train"],
            "validation": load_dataset("json", data_files=files_val)["train"]
        })

        # Subsample train per strategy config, leave validation intact
        if 0 < self.training_subset_fraction < 1.0:
            n = int(len(ds["train"]) * self.training_subset_fraction)
            ds["train"] = ds["train"].select(range(n))

        # --- CRITICAL: drop ALL original columns so the collator only sees tensors ---
        orig_cols = ds["train"].column_names  # e.g., ['id','document_id','text','tokens','ner_tags']
        ds_tok = ds.map(self._encode_align, batched=True, remove_columns=orig_cols)

        trainloader = DataLoader(
            ds_tok["train"], batch_size=self.batch_size, shuffle=True, collate_fn=self.collator
        )
        valloader   = DataLoader(
            ds_tok["validation"], batch_size=self.batch_size, shuffle=False, collate_fn=self.collator
        )
        return trainloader, valloader

    def load_datasets(self):
        trainloaders, valloaders = [], []
        for client_folder in sorted(os.listdir(self.dataset_dir), key=lambda s: int(s.split("_")[1])):
            if client_folder.startswith("."):
                continue
            cdir = os.path.join(self.dataset_dir, client_folder)
            tr, va = self._load_one_client(cdir)
            trainloaders.append(tr)
            valloaders.append(va)
        return trainloaders, valloaders
