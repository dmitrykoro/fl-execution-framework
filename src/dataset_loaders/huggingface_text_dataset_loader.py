import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


class HuggingFaceTextDatasetLoader:
    """
    Unified dataset loader for HuggingFace text datasets with MLM task support.

    Examples:
        Financial PhraseBank: hf_dataset_path="takala/financial_phrasebank",
                             hf_dataset_name="sentences_allagree",
                             tokenize_columns=["sentence"]
        LexGLUE LEDGAR: hf_dataset_path="coastalcph/lex_glue",
                       hf_dataset_name="ledgar",
                       tokenize_columns=["text"]
    """

    def __init__(
        self,
        hf_dataset_path: str,
        hf_dataset_name: str = None,
        tokenize_columns: list = None,
        remove_columns: list = None,
        dataset_dir: str = None,  # Not used, kept for compatibility
        num_of_clients: int = 5,
        training_subset_fraction: float = 0.8,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 16,
        chunk_size: int = 256,
        mlm_probability: float = 0.15,
        num_poisoned_clients: int = 0,
        attack_schedule=None,
    ):
        self.hf_dataset_path = hf_dataset_path
        self.hf_dataset_name = hf_dataset_name
        self.tokenize_columns = tokenize_columns or ["text"]
        self.remove_columns = remove_columns
        self.num_of_clients = num_of_clients
        self.training_subset_fraction = training_subset_fraction
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.mlm_probability = mlm_probability
        self.num_poisoned_clients = num_poisoned_clients
        self.attack_schedule = attack_schedule
        self.tokenizer = None

    def _partition_iid(self, full_dataset):
        """Partition dataset uniformly across clients (IID distribution)."""
        client_size = len(full_dataset) // self.num_of_clients
        client_indices = []

        for client_id in range(self.num_of_clients):
            start_idx = client_id * client_size
            end_idx = start_idx + client_size if client_id < self.num_of_clients - 1 else len(full_dataset)
            client_indices.append(list(range(start_idx, end_idx)))

        return client_indices

    def _partition_label_skew_dirichlet(self, full_dataset, alpha=0.5):
        """
        Partition dataset using Dirichlet distribution (Non-IID).
        Lower alpha = more heterogeneous (typical: 0.1-1.0).
        """
        labels = np.array(full_dataset["label"])
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(self.num_of_clients)]

        # Use fixed seed for reproducibility
        rng = np.random.default_rng(42)

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            proportions = rng.dirichlet(np.repeat(alpha, self.num_of_clients))
            proportions = (proportions * len(class_indices)).astype(int)

            # Adjust to ensure all samples are assigned
            proportions[-1] = len(class_indices) - proportions[:-1].sum()

            rng.shuffle(class_indices)

            start_idx = 0
            for client_id, count in enumerate(proportions):
                end_idx = start_idx + count
                client_indices[client_id].extend(class_indices[start_idx:end_idx].tolist())
                start_idx = end_idx

        # Shuffle each client's indices to mix classes
        for indices in client_indices:
            rng.shuffle(indices)

        return client_indices

    def load_datasets(self):
        """Loads dataset from HuggingFace Hub and partitions into clients."""
        trainloaders = []
        valloaders = []

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer = self.tokenizer

        # Skip filesystem poisoning (dynamic poisoning handled in training)
        if self.attack_schedule:
            poisoned_client_ids = []
        else:
            poisoned_client_ids = list(range(self.num_poisoned_clients))

        if self.hf_dataset_name:
            dataset = load_dataset(self.hf_dataset_path, self.hf_dataset_name, trust_remote_code=True)
        else:
            dataset = load_dataset(self.hf_dataset_path, trust_remote_code=True)

        full_dataset = dataset["train"]

        if self.remove_columns is None:
            self.remove_columns = [col for col in full_dataset.column_names if col not in ["input_ids", "attention_mask"]]

        # Use Non-IID for labeled datasets, IID for unlabeled
        if "label" in full_dataset.column_names:
            client_indices_list = self._partition_label_skew_dirichlet(full_dataset, alpha=0.5)
        else:
            full_dataset = full_dataset.shuffle(seed=42)
            client_indices_list = self._partition_iid(full_dataset)

        for client_id in range(self.num_of_clients):
            client_dataset = full_dataset.select(client_indices_list[client_id])

            def tokenize_function(examples):
                texts = [" ".join(row) for row in zip(*[examples[col] for col in self.tokenize_columns])]
                return tokenizer(texts, truncation=False)

            def chunk_function(examples):
                concatenated = {k: sum(examples[k], []) for k in examples.keys()}
                total_len = len(concatenated["input_ids"])
                total_len = (total_len // self.chunk_size) * self.chunk_size

                result = {
                    k: [t[i:i + self.chunk_size] for i in range(0, total_len, self.chunk_size)]
                    for k, t in concatenated.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            client_dataset = client_dataset.map(tokenize_function, batched=True)
            columns_to_remove = [col for col in self.remove_columns if col in client_dataset.column_names]
            if columns_to_remove:
                client_dataset = client_dataset.remove_columns(columns_to_remove)
            client_dataset = client_dataset.map(chunk_function, batched=True)
            split_dataset = client_dataset.train_test_split(test_size=(1 - self.training_subset_fraction), seed=42)

            collate_fn = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=.75 if client_id in poisoned_client_ids else self.mlm_probability,
                mask_replace_prob=0 if client_id in poisoned_client_ids else 0.8,
                random_replace_prob=1 if client_id in poisoned_client_ids else 0.1,
            )

            trainloader = DataLoader(
                split_dataset["train"],
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,  # Avoid CUDA fork issues
                pin_memory=torch.cuda.is_available()  # Fast GPU transfer
            )
            valloader = DataLoader(
                split_dataset["test"],
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            trainloaders.append(trainloader)
            valloaders.append(valloader)

        return trainloaders, valloaders
