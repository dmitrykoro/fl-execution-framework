import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

class FinancialPhraseBankDatasetLoader:
    """
    Dataset loader for FinancialPhraseBank (HuggingFace: takala/financial_phrasebank).

    Dataset: 4,840 sentences from English financial news categorized by sentiment.
    Partitions data into N clients for federated learning with MLM task.
    """

    def __init__(self,
            dataset_dir: str = None,  # Not used, loads from HuggingFace
            num_of_clients: int = 5,
            training_subset_fraction: float = 0.8,
            model_name: str = "distilbert-base-uncased",
            batch_size: int = 16,
            chunk_size: int = 256,
            mlm_probability: float = 0.15,
            num_poisoned_clients: int = 0,
            attack_schedule=None,
            tokenize_columns=["sentence"],
            remove_columns=["sentence", "label"],
        ):
        self.num_of_clients = num_of_clients
        self.training_subset_fraction = training_subset_fraction
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.mlm_probability = mlm_probability
        self.num_poisoned_clients = num_poisoned_clients
        self.attack_schedule = attack_schedule
        self.tokenize_columns = tokenize_columns
        self.remove_columns = remove_columns
        self.tokenizer = None

    def load_datasets(self):
        """
        Loads FinancialPhraseBank from HuggingFace and partitions into clients.
        """
        trainloaders = []
        valloaders = []

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer = self.tokenizer

        # Skip filesystem poisoning (dynamic poisoning handled in training)
        if self.attack_schedule:
            poisoned_client_ids = []
        else:
            poisoned_client_ids = list(range(self.num_poisoned_clients))

        # Load dataset from HuggingFace
        dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        full_dataset = dataset["train"]

        # Partition into N clients (simple splitting)
        client_size = len(full_dataset) // self.num_of_clients

        for client_id in range(self.num_of_clients):
            start_idx = client_id * client_size
            end_idx = start_idx + client_size if client_id < self.num_of_clients - 1 else len(full_dataset)

            client_dataset = full_dataset.select(range(start_idx, end_idx))

            # Tokenize text
            def tokenize_function(examples):
                texts = [" ".join(row) for row in zip(*[examples[col] for col in self.tokenize_columns])]
                return tokenizer(texts, truncation=False)

            # Chunk tokens into fixed-length blocks
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
            split_dataset = client_dataset.train_test_split(test_size=(1 - self.training_subset_fraction))

            # DataLoader preparation
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
