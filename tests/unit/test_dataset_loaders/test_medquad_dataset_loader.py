from unittest.mock import patch

from src.dataset_loaders.medquad_dataset_loader import MedQuADDatasetLoader
from tests.common import Mock, pytest


class TestMedQuADDatasetLoader:
    @pytest.fixture
    def temp_dataset_dir(self, tmp_path):
        """Return a temporary MedQuAD-style dataset with JSON files for clients."""
        dataset_dir = tmp_path / "medquad_dataset"
        dataset_dir.mkdir()

        for i in range(3):
            client_dir = dataset_dir / f"client_{i}"
            client_dir.mkdir()
            json_file = client_dir / f"data_{i}.json"
            json_file.write_text(
                '{"question": "What is test?", "answer": "This is a test answer"}'
            )

        return str(dataset_dir)

    @pytest.fixture
    def dataset_loader(self, temp_dataset_dir):
        """Return a MedQuADDatasetLoader configured for tests."""
        return MedQuADDatasetLoader(
            dataset_dir=temp_dataset_dir,
            num_of_clients=3,
            training_subset_fraction=0.8,
            model_name="bert-base-uncased",
            batch_size=16,
            chunk_size=256,
            mlm_probability=0.15,
            num_poisoned_clients=1,
        )

    def test_init_sets_attributes_correctly(self, temp_dataset_dir):
        """Verify MedQuADDatasetLoader.__init__ sets expected attributes."""
        loader = MedQuADDatasetLoader(
            dataset_dir=temp_dataset_dir,
            num_of_clients=5,
            training_subset_fraction=0.7,
            model_name="distilbert-base-uncased",
            batch_size=32,
            chunk_size=512,
            mlm_probability=0.2,
            num_poisoned_clients=2,
            tokenize_columns=["question", "answer"],
            remove_columns=["question", "answer"],
        )

        assert loader.dataset_dir == temp_dataset_dir
        assert loader.num_of_clients == 5
        assert loader.training_subset_fraction == 0.7
        assert loader.model_name == "distilbert-base-uncased"
        assert loader.batch_size == 32
        assert loader.chunk_size == 512
        assert loader.mlm_probability == 0.2
        assert loader.num_poisoned_clients == 2
        assert loader.tokenize_columns == ["question", "answer"]
        assert loader.remove_columns == ["question", "answer"]

    def test_init_with_default_parameters(self, temp_dataset_dir):
        """Verify default parameter values are applied when omitted."""
        loader = MedQuADDatasetLoader(
            dataset_dir=temp_dataset_dir,
            num_of_clients=3,
            training_subset_fraction=0.8,
            model_name="bert-base-uncased",
        )

        # Check default values
        assert loader.batch_size == 16
        assert loader.chunk_size == 256
        assert loader.mlm_probability == 0.15
        assert loader.num_poisoned_clients == 0
        assert loader.tokenize_columns == ["answer"]
        assert loader.remove_columns == ["answer", "token_type_ids", "question"]

    @patch("src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained")
    @patch("src.dataset_loaders.medquad_dataset_loader.load_dataset")
    @patch("src.dataset_loaders.medquad_dataset_loader.DataLoader")
    @patch("src.dataset_loaders.medquad_dataset_loader.glob.glob")
    def test_load_datasets_processes_client_folders(
        self,
        mock_glob,
        mock_dataloader,
        mock_load_dataset,
        mock_tokenizer,
        dataset_loader,
    ):
        """Verify load_datasets loads datasets and returns train/val loaders per client."""
        # Mock glob to return JSON files
        mock_glob.return_value = ["client_0/data.json"]

        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Create mock DatasetDict that supports the chain of operations
        mock_dataset_dict = Mock()
        mock_train_dataset = Mock()

        # Configure the mock chain: DatasetDict.map() -> DatasetDict.remove_columns() -> DatasetDict.map() -> DatasetDict["train"] -> Dataset.train_test_split()
        mock_dataset_dict.map.return_value = mock_dataset_dict
        mock_dataset_dict.remove_columns.return_value = mock_dataset_dict
        mock_dataset_dict.__getitem__ = Mock(return_value=mock_train_dataset)

        mock_train_dataset.train_test_split.return_value = {
            "train": Mock(),
            "test": Mock(),
        }

        # load_dataset returns the DatasetDict
        mock_load_dataset.return_value = mock_dataset_dict

        # Mock DataLoader (need 2 loaders per client folder, 3 folders = 6 total)
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        mock_dataloader.side_effect = [
            mock_train_loader,
            mock_val_loader,  # client_0
            mock_train_loader,
            mock_val_loader,  # client_1
            mock_train_loader,
            mock_val_loader,  # client_2
        ]

        trainloaders, valloaders = dataset_loader.load_datasets()

        # Should create tokenizer
        mock_tokenizer.assert_called_once_with("bert-base-uncased")

        # Should load dataset for each client
        assert mock_load_dataset.call_count == 3

        # Should return lists of loaders
        assert len(trainloaders) == 3
        assert len(valloaders) == 3

    @patch("src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained")
    @patch("src.dataset_loaders.medquad_dataset_loader.load_dataset")
    @patch("src.dataset_loaders.medquad_dataset_loader.DataLoader")
    @patch("src.dataset_loaders.medquad_dataset_loader.glob.glob")
    def test_load_datasets_handles_poisoned_clients(
        self,
        mock_glob,
        mock_dataloader,
        mock_load_dataset,
        mock_tokenizer,
        dataset_loader,
    ):
        """Verify poisoned clients receive the configured MLM collator settings."""
        # Mock components
        mock_glob.return_value = ["data.json"]
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Create mock DatasetDict that supports the chain of operations
        mock_dataset_dict = Mock()
        mock_train_dataset = Mock()

        # Configure the mock chain
        mock_dataset_dict.map.return_value = mock_dataset_dict
        mock_dataset_dict.remove_columns.return_value = mock_dataset_dict
        mock_dataset_dict.__getitem__ = Mock(return_value=mock_train_dataset)

        mock_train_dataset.train_test_split.return_value = {
            "train": Mock(),
            "test": Mock(),
        }

        mock_load_dataset.return_value = mock_dataset_dict

        with patch(
            "src.dataset_loaders.medquad_dataset_loader.DataCollatorForLanguageModeling"
        ) as mock_collator:
            mock_collator_instance = Mock()
            mock_collator.return_value = mock_collator_instance

            dataset_loader.load_datasets()

            # Check that different MLM probabilities are used
            collator_calls = mock_collator.call_args_list

            # First client (client_0) should be poisoned
            poisoned_call = collator_calls[0]
            args, kwargs = poisoned_call
            assert kwargs["mlm_probability"] == 0.75  # Poisoned client

            # Second client (client_1) should be normal
            normal_call = collator_calls[1]
            args, kwargs = normal_call
            assert kwargs["mlm_probability"] == 0.15  # Normal client

    @patch("src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained")
    @patch("src.dataset_loaders.medquad_dataset_loader.load_dataset")
    def test_load_datasets_applies_tokenization(
        self, mock_load_dataset, mock_tokenizer, dataset_loader
    ):
        """Verify tokenization and column removal are applied to loaded datasets."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Create mock DatasetDict that supports the chain of operations
        mock_dataset_dict = Mock()
        mock_train_dataset = Mock()

        # Configure the mock chain: DatasetDict.map() -> DatasetDict.remove_columns() -> DatasetDict.map() -> DatasetDict["train"] -> Dataset.train_test_split()
        mock_dataset_dict.map.return_value = mock_dataset_dict
        mock_dataset_dict.remove_columns.return_value = mock_dataset_dict
        mock_dataset_dict.__getitem__ = Mock(return_value=mock_train_dataset)

        mock_train_dataset.train_test_split.return_value = {
            "train": Mock(),
            "test": Mock(),
        }

        mock_load_dataset.return_value = mock_dataset_dict

        with patch(
            "src.dataset_loaders.medquad_dataset_loader.glob.glob",
            return_value=["data.json"],
        ):
            with patch("src.dataset_loaders.medquad_dataset_loader.DataLoader"):
                with patch(
                    "src.dataset_loaders.medquad_dataset_loader.DataCollatorForLanguageModeling"
                ):
                    dataset_loader.load_datasets()

        # Should apply tokenization mapping
        assert mock_dataset_dict.map.called
        # Should remove specified columns
        assert mock_dataset_dict.remove_columns.called
        # Should apply chunking mapping (2 map calls per client, 3 clients = 6 total)
        assert mock_dataset_dict.map.call_count == 6

    def test_load_datasets_skips_hidden_files(self, dataset_loader):
        """Verify hidden files/folders are ignored when scanning client folders."""
        with patch(
            "src.dataset_loaders.medquad_dataset_loader.os.listdir"
        ) as mock_listdir:
            mock_listdir.return_value = ["client_0", ".DS_Store", "client_1"]

            with patch(
                "src.dataset_loaders.medquad_dataset_loader.glob.glob",
                return_value=["data.json"],
            ):
                with patch(
                    "src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained"
                ):
                    with patch(
                        "src.dataset_loaders.medquad_dataset_loader.load_dataset"
                    ) as mock_load_dataset:
                        # Create mock DatasetDict that supports the chain of operations
                        mock_dataset_dict = Mock()
                        mock_train_dataset = Mock()

                        # Configure the mock chain
                        mock_dataset_dict.map.return_value = mock_dataset_dict
                        mock_dataset_dict.remove_columns.return_value = (
                            mock_dataset_dict
                        )
                        mock_dataset_dict.__getitem__ = Mock(
                            return_value=mock_train_dataset
                        )

                        mock_train_dataset.train_test_split.return_value = {
                            "train": Mock(),
                            "test": Mock(),
                        }

                        mock_load_dataset.return_value = mock_dataset_dict

                        with patch(
                            "src.dataset_loaders.medquad_dataset_loader.DataLoader"
                        ):
                            with patch(
                                "src.dataset_loaders.medquad_dataset_loader.DataCollatorForLanguageModeling"
                            ):
                                trainloaders, valloaders = (
                                    dataset_loader.load_datasets()
                                )

        # Should only process non-hidden folders (client_0 and client_1)
        assert mock_load_dataset.call_count == 2

    @patch("src.dataset_loaders.medquad_dataset_loader.os.listdir")
    def test_load_datasets_sorts_client_folders_correctly(
        self, mock_listdir, dataset_loader
    ):
        """Verify client folders are processed in numeric order by suffix."""
        mock_listdir.return_value = ["client_10", "client_2", "client_1"]

        with patch(
            "src.dataset_loaders.medquad_dataset_loader.glob.glob",
            return_value=["data.json"],
        ):
            with patch(
                "src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained"
            ):
                with patch(
                    "src.dataset_loaders.medquad_dataset_loader.load_dataset"
                ) as mock_load_dataset:
                    # Create mock DatasetDict that supports the chain of operations
                    mock_dataset_dict = Mock()
                    mock_train_dataset = Mock()

                    # Configure the mock chain
                    mock_dataset_dict.map.return_value = mock_dataset_dict
                    mock_dataset_dict.remove_columns.return_value = mock_dataset_dict
                    mock_dataset_dict.__getitem__ = Mock(
                        return_value=mock_train_dataset
                    )

                    mock_train_dataset.train_test_split.return_value = {
                        "train": Mock(),
                        "test": Mock(),
                    }

                    mock_load_dataset.return_value = mock_dataset_dict

                    with patch("src.dataset_loaders.medquad_dataset_loader.DataLoader"):
                        with patch(
                            "src.dataset_loaders.medquad_dataset_loader.DataCollatorForLanguageModeling"
                        ):
                            dataset_loader.load_datasets()

        # Should be called 3 times in sorted order
        assert mock_load_dataset.call_count == 3

    def test_load_datasets_creates_correct_poisoned_client_list(self, dataset_loader):
        """Verify the loader identifies the correct client IDs to poison."""
        # Set num_poisoned_clients to 2
        dataset_loader.num_poisoned_clients = 2

        with patch(
            "src.dataset_loaders.medquad_dataset_loader.os.listdir"
        ) as mock_listdir:
            mock_listdir.return_value = ["client_0", "client_1", "client_2"]

            with patch(
                "src.dataset_loaders.medquad_dataset_loader.glob.glob",
                return_value=["data.json"],
            ):
                with patch(
                    "src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained"
                ):
                    with patch(
                        "src.dataset_loaders.medquad_dataset_loader.load_dataset"
                    ) as mock_load_dataset:
                        # Create mock DatasetDict that supports the chain of operations
                        mock_dataset_dict = Mock()
                        mock_train_dataset = Mock()

                        # Configure the mock chain
                        mock_dataset_dict.map.return_value = mock_dataset_dict
                        mock_dataset_dict.remove_columns.return_value = (
                            mock_dataset_dict
                        )
                        mock_dataset_dict.__getitem__ = Mock(
                            return_value=mock_train_dataset
                        )

                        mock_train_dataset.train_test_split.return_value = {
                            "train": Mock(),
                            "test": Mock(),
                        }

                        mock_load_dataset.return_value = mock_dataset_dict

                        with patch(
                            "src.dataset_loaders.medquad_dataset_loader.DataLoader"
                        ):
                            with patch(
                                "src.dataset_loaders.medquad_dataset_loader.DataCollatorForLanguageModeling"
                            ) as mock_collator:
                                dataset_loader.load_datasets()

                                # Check that first 2 clients get poisoned settings
                                collator_calls = mock_collator.call_args_list

                                # Client 0 and 1 should be poisoned
                                for i in [0, 1]:
                                    args, kwargs = collator_calls[i]
                                    assert kwargs["mlm_probability"] == 0.75
                                    assert kwargs["mask_replace_prob"] == 0
                                    assert kwargs["random_replace_prob"] == 1

                                # Client 2 should be normal
                                args, kwargs = collator_calls[2]
                                assert kwargs["mlm_probability"] == 0.15
                                assert kwargs["mask_replace_prob"] == 0.8
                                assert kwargs["random_replace_prob"] == 0.1

    def test_tokenize_function_joins_columns_correctly(self, dataset_loader):
        """Test internal tokenize_function joins specified columns correctly"""
        with patch(
            "src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer:
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3]}
            mock_tokenizer.return_value = mock_tokenizer_instance

            # Create a tokenize function like the one in load_datasets
            tokenizer = mock_tokenizer_instance
            tokenize_columns = ["answer"]

            def tokenize_function(examples):
                texts = [
                    " ".join(row)
                    for row in zip(*[examples[col] for col in tokenize_columns])
                ]
                return tokenizer(texts, truncation=False)

            # Test with sample data
            examples = {"answer": ["This is answer 1", "This is answer 2"]}
            tokenize_function(examples)

            # Should call tokenizer with joined texts
            mock_tokenizer_instance.assert_called_with(
                ["This is answer 1", "This is answer 2"], truncation=False
            )

    def test_tokenize_function_with_multiple_columns(self, dataset_loader):
        """Test tokenize_function joins multiple columns with spaces"""
        with patch(
            "src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer:
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3]}
            mock_tokenizer.return_value = mock_tokenizer_instance

            # Create tokenize function with multiple columns
            tokenizer = mock_tokenizer_instance
            tokenize_columns = ["question", "answer"]

            def tokenize_function(examples):
                texts = [
                    " ".join(row)
                    for row in zip(*[examples[col] for col in tokenize_columns])
                ]
                return tokenizer(texts, truncation=False)

            # Test with sample data
            examples = {
                "question": ["What is AI?", "What is ML?"],
                "answer": ["Artificial Intelligence", "Machine Learning"],
            }
            tokenize_function(examples)

            # Should join question and answer with space
            mock_tokenizer_instance.assert_called_with(
                [
                    "What is AI? Artificial Intelligence",
                    "What is ML? Machine Learning",
                ],
                truncation=False,
            )

    def test_chunk_function_chunks_tokens_correctly(self, dataset_loader):
        """Test internal chunk_function splits tokens into fixed-size chunks"""
        chunk_size = 256

        def chunk_function(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_len = len(concatenated["input_ids"])
            total_len = (total_len // chunk_size) * chunk_size

            result = {
                k: [t[i : i + chunk_size] for i in range(0, total_len, chunk_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Test with sample tokenized data (3 examples with varying lengths)
        examples = {
            "input_ids": [
                list(range(100)),  # 100 tokens
                list(range(100, 250)),  # 150 tokens
                list(range(250, 400)),  # 150 tokens
            ],
            "attention_mask": [
                [1] * 100,
                [1] * 150,
                [1] * 150,
            ],
        }

        result = chunk_function(examples)

        # Total tokens: 100 + 150 + 150 = 400
        # With chunk_size=256, we get: floor(400/256) = 1 chunk
        # Expected: 1 chunk of 256 tokens (remaining 144 tokens are dropped)
        assert len(result["input_ids"]) == 1
        assert len(result["input_ids"][0]) == 256
        assert len(result["attention_mask"]) == 1
        assert len(result["attention_mask"][0]) == 256

        # Should create labels as copy of input_ids
        assert "labels" in result
        assert result["labels"] == result["input_ids"]

    def test_chunk_function_handles_exact_multiple_of_chunk_size(self, dataset_loader):
        """Test chunk_function when total length is exact multiple of chunk_size"""
        chunk_size = 128

        def chunk_function(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_len = len(concatenated["input_ids"])
            total_len = (total_len // chunk_size) * chunk_size

            result = {
                k: [t[i : i + chunk_size] for i in range(0, total_len, chunk_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Test with data that's exactly 2 * chunk_size
        examples = {
            "input_ids": [
                list(range(128)),  # 128 tokens
                list(range(128, 256)),  # 128 tokens
            ],
            "attention_mask": [
                [1] * 128,
                [1] * 128,
            ],
        }

        result = chunk_function(examples)

        # Should create exactly 2 chunks
        assert len(result["input_ids"]) == 2
        assert len(result["input_ids"][0]) == 128
        assert len(result["input_ids"][1]) == 128

    def test_chunk_function_drops_incomplete_chunk(self, dataset_loader):
        """Test chunk_function drops tokens that don't fill a complete chunk"""
        chunk_size = 100

        def chunk_function(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_len = len(concatenated["input_ids"])
            total_len = (total_len // chunk_size) * chunk_size

            result = {
                k: [t[i : i + chunk_size] for i in range(0, total_len, chunk_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Test with 250 tokens (2 complete chunks + 50 leftover)
        examples = {
            "input_ids": [list(range(250))],
            "attention_mask": [[1] * 250],
        }

        result = chunk_function(examples)

        # Should only create 2 complete chunks (50 tokens dropped)
        assert len(result["input_ids"]) == 2
        assert len(result["input_ids"][0]) == 100
        assert len(result["input_ids"][1]) == 100

    def test_chunk_function_creates_labels_copy(self, dataset_loader):
        """Test chunk_function creates independent labels copy"""
        chunk_size = 64

        def chunk_function(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_len = len(concatenated["input_ids"])
            total_len = (total_len // chunk_size) * chunk_size

            result = {
                k: [t[i : i + chunk_size] for i in range(0, total_len, chunk_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        examples = {
            "input_ids": [list(range(128))],
            "attention_mask": [[1] * 128],
        }

        result = chunk_function(examples)

        # Labels should be a copy, not the same reference
        assert result["labels"] == result["input_ids"]
        assert result["labels"] is not result["input_ids"]

        # Verify both have the expected structure
        assert len(result["labels"]) == 2
        assert len(result["labels"][0]) == 64
        assert len(result["labels"][1]) == 64

    @patch("src.dataset_loaders.medquad_dataset_loader.AutoTokenizer.from_pretrained")
    @patch("src.dataset_loaders.medquad_dataset_loader.load_dataset")
    @patch("src.dataset_loaders.medquad_dataset_loader.DataLoader")
    @patch("src.dataset_loaders.medquad_dataset_loader.glob.glob")
    def test_load_datasets_uses_correct_train_test_split(
        self,
        mock_glob,
        mock_dataloader,
        mock_load_dataset,
        mock_tokenizer,
        dataset_loader,
    ):
        """Test load_datasets uses correct train/test split fraction"""
        mock_glob.return_value = ["data.json"]
        mock_tokenizer.return_value = Mock()

        # Create mock DatasetDict that supports the chain of operations
        mock_dataset_dict = Mock()
        mock_train_dataset = Mock()

        # Configure the mock chain
        mock_dataset_dict.map.return_value = mock_dataset_dict
        mock_dataset_dict.remove_columns.return_value = mock_dataset_dict
        mock_dataset_dict.__getitem__ = Mock(return_value=mock_train_dataset)

        mock_train_dataset.train_test_split.return_value = {
            "train": Mock(),
            "test": Mock(),
        }

        mock_load_dataset.return_value = mock_dataset_dict

        with patch(
            "src.dataset_loaders.medquad_dataset_loader.DataCollatorForLanguageModeling"
        ):
            dataset_loader.load_datasets()

        # Should use correct test_size (1 - training_subset_fraction)
        expected_test_size = 1 - dataset_loader.training_subset_fraction
        mock_train_dataset.train_test_split.assert_called_with(
            test_size=expected_test_size
        )
