# Configuration Quick Start

> Copy-paste these templates and modify as needed. Each template is ready to run immediately.

**Simulation Entry Point**: `src/simulation_runner.py` - Main execution script that loads configs and runs experiments

## Table of Contents

- [Working Templates](#working-templates)
- [Example Config Files](#-example-configurations)
- [Next Steps](#-next-steps)

---

## Working Templates

### 1. Basic Training (No Attacks)

**Use case**: Test baseline performance, understand data distribution, validate setup

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "trust",
    "dataset_keyword": "femnist_iid",
    "num_of_rounds": 20,
    "num_of_clients": 10,
    "num_of_malicious_clients": 0,
    "attack_schedule": [],
    "strict_mode": "true",
    "remove_clients": "false",

    "show_plots": "true",
    "save_plots": "true",
    "save_csv": "true",
    "preserve_dataset": "false",
    "training_subset_fraction": 0.9,
    "model_type": "cnn",

    "training_device": "cpu",
    "cpus_per_client": 2,
    "gpus_per_client": 0,
    "num_of_client_epochs": 1,
    "batch_size": 32,

    "trust_threshold": 0.15,
    "beta_value": 0.75,
    "num_of_clusters": 1,
    "begin_removing_from_round": 4
  },
  "simulation_strategies": [{}]
}
```

---

### 2. Label Flipping Attack Detection

**Use case**: Test Byzantine-tolerant strategies against simple adversaries

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "multi-krum",
    "dataset_keyword": "pneumoniamnist",
    "num_of_rounds": 30,
    "num_of_clients": 10,
    "num_of_malicious_clients": 2,
    "attack_schedule": [
      {
        "attack_type": "label_flipping",
        "flip_fraction": 1.0,
        "start_round": 1,
        "end_round": 30,
        "selection_strategy": "percentage",
        "malicious_percentage": 0.2
      }
    ],
    "strict_mode": "true",
    "remove_clients": "true",

    "show_plots": "true",
    "save_plots": "true",
    "save_csv": "true",
    "preserve_dataset": "false",
    "training_subset_fraction": 0.9,
    "model_type": "cnn",

    "training_device": "cpu",
    "cpus_per_client": 2,
    "gpus_per_client": 0,
    "num_of_client_epochs": 1,
    "batch_size": 32,

    "num_krum_selections": 7
  },
  "simulation_strategies": [{}]
}
```

---

### 3. Gaussian Noise Attack (Advanced)

**Use case**: Test robustness against data poisoning

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "rfa",
    "dataset_keyword": "bloodmnist",
    "num_of_rounds": 25,
    "num_of_clients": 10,
    "num_of_malicious_clients": 3,
    "attack_schedule": [
      {
        "attack_type": "gaussian_noise",
        "target_noise_snr": 5.0,
        "attack_ratio": 0.5,
        "start_round": 1,
        "end_round": 25,
        "selection_strategy": "percentage",
        "malicious_percentage": 0.3
      }
    ],
    "strict_mode": "true",
    "remove_clients": "false",

    "show_plots": "true",
    "save_plots": "true",
    "save_csv": "true",
    "preserve_dataset": "true",
    "training_subset_fraction": 0.9,
    "model_type": "cnn",

    "training_device": "cpu",
    "cpus_per_client": 2,
    "gpus_per_client": 0,
    "num_of_client_epochs": 1,
    "batch_size": 32
  },
  "simulation_strategies": [{}]
}
```

---

### 4. LLM Fine-Tuning with LoRA

**Use case**: Federated learning for language models

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "trust",
    "dataset_keyword": "medquad",
    "num_of_rounds": 15,
    "num_of_clients": 5,
    "num_of_malicious_clients": 0,
    "attack_schedule": [],
    "strict_mode": "true",
    "remove_clients": "false",

    "show_plots": "true",
    "save_plots": "true",
    "save_csv": "true",
    "preserve_dataset": "false",
    "training_subset_fraction": 0.9,
    "model_type": "transformer",

    "use_llm": "true",
    "llm_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "llm_finetuning": "lora",
    "llm_task": "mlm",
    "llm_chunk_size": 128,
    "mlm_probability": 0.15,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "value"],

    "training_device": "cpu",
    "cpus_per_client": 2,
    "gpus_per_client": 0,
    "num_of_client_epochs": 2,
    "batch_size": 8,

    "trust_threshold": 0.15,
    "beta_value": 0.75,
    "num_of_clusters": 1,
    "begin_removing_from_round": 4
  },
  "simulation_strategies": [{}]
}
```

---

### 5. Strategy Comparison

**Use case**: Compare multiple aggregation strategies side-by-side

```json
{
  "shared_settings": {
    "dataset_keyword": "femnist_iid",
    "num_of_rounds": 25,
    "num_of_clients": 10,
    "num_of_malicious_clients": 2,
    "attack_schedule": [
      {
        "attack_type": "label_flipping",
        "flip_fraction": 1.0,
        "start_round": 1,
        "end_round": 25,
        "selection_strategy": "percentage",
        "malicious_percentage": 0.2
      }
    ],
    "strict_mode": "true",
    "remove_clients": "true",

    "show_plots": "true",
    "save_plots": "true",
    "save_csv": "true",
    "preserve_dataset": "false",
    "training_subset_fraction": 0.9,
    "model_type": "cnn",

    "training_device": "cpu",
    "cpus_per_client": 2,
    "gpus_per_client": 0,
    "num_of_client_epochs": 1,
    "batch_size": 32,

    "num_krum_selections": 7,
    "trim_ratio": 0.2,
    "num_std_dev": 2.0,
    "Kp": 1.0,
    "Ki": 0.1,
    "Kd": 0.05
  },
  "simulation_strategies": [
    {
      "aggregation_strategy_keyword": "multi-krum"
    },
    {
      "aggregation_strategy_keyword": "trimmed_mean"
    },
    {
      "aggregation_strategy_keyword": "bulyan"
    },
    {
      "aggregation_strategy_keyword": "pid_standardized"
    }
  ]
}
```

---

## 📦 Example Configurations

The framework includes working examples in `config/simulation_strategies/examples/`:

| Config File | Purpose | Key Features |
|------------|---------|--------------|
| `dynamic_poisoning_config.json` | Attack stacking demo | Label flipping + Gaussian noise attacks |
| `brightness_attack_config.json` | Image manipulation | Brightness poisoning attack |
| `token_replacement_config.json` | NLP attacks | Token-level poisoning for transformers |
| `bulyan_config.json` | Byzantine defense | Bulyan aggregation strategy |
| `krum_testing_config.json` | Krum validation | Multi-Krum strategy testing |
| `mkrum_config.json` | Multi-Krum demo | Multiple client selection |
| `its_config.json` | Traffic sign dataset | Non-IID data distribution |
| `medquad_hf_config.json` | LLM fine-tuning | LoRA + BERT for medical Q&A |
| `attack_visualization_config.json` | Visual snapshots | PNG + JSON snapshot demo |
| `shading_test_config.json` | Plot shading visualization | 4 overlapping attack types with background shading |

**Quick start:**

```bash
python -m src.simulation_runner examples/bulyan_config.json
```

**Tip**: Copy an example config as a starting point for your experiments!

```bash
cp config/simulation_strategies/examples/bulyan_config.json config/simulation_strategies/my_experiment.json
# Edit my_experiment.json with your parameters
python -m src.simulation_runner my_experiment.json
```

---

## 📚 Next Steps

- **Understand parameters**: See [Parameter Reference](parameters.md) for detailed explanation of all configuration options
- **Choose strategies**: See [Strategy Guide](strategies.md) for help selecting the right aggregation strategy
- **Having issues?** Check [Troubleshooting Guide](troubleshooting.md) for common errors and fixes
- **Attack scheduling**: See [Attack Scheduling Guide](../attack-scheduling.md) for round-based poisoning
- **Main README**: Back to [README](../../README.md)
