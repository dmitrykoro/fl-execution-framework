import os
import glob
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

class MedQuADDatasetLoader:
    def __init__(self,
            dataset_dir: str,
            num_of_clients: int,
            training_subset_fraction: float,
            model_name: str,
            batch_size: int = 16,
            chunk_size: int = 256,
            mlm_probability: float = 0.15,
            tokenize_columns=["answer"],
            remove_columns=["answer", "token_type_ids", "question"],
        ):
        self.dataset_dir = dataset_dir
        self.num_of_clients = num_of_clients
        self.training_subset_fraction = training_subset_fraction
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.mlm_probability = mlm_probability
        self.tokenize_columns = tokenize_columns
        self.remove_columns = remove_columns
    
    def load_datasets(self):
        """
        Loads and tokenizes dataset for masked language modeling (MLM).
        """
        
        trainloaders = []
        valloaders = []

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        for client_folder in sorted(os.listdir(self.dataset_dir), key=lambda string: int(string.split("_")[1])):
            if client_folder.startswith("."):  # .DS_store
                continue

            json_files = glob.glob(os.path.join(self.dataset_dir, client_folder, "*.json"))

            client_dataset = load_dataset("json", data_files=json_files)

            # Tokenize answers
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
            client_dataset = client_dataset.remove_columns(self.remove_columns)
            client_dataset = client_dataset.map(chunk_function, batched=True)
            dataset = client_dataset["train"].train_test_split(test_size=(1 - self.training_subset_fraction))

            # DataLoader preparation
            collate_fn = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability,
            )

            trainloader = DataLoader(
                dataset["train"], batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
            )
            valloader = DataLoader(
                dataset["test"], batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
            )


            trainloaders.append(trainloader)
            valloaders.append(valloader)

        return trainloaders, valloaders
