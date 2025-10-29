# Federated Learning Simulation Framework

> *Framework for simulating, testing, and optimizing federated learning strategies*  

[![codecov](https://codecov.io/github/dmitrykoro/fl-execution-framework/graph/badge.svg?token=HJFASRJ43T)](https://codecov.io/github/dmitrykoro/fl-execution-framework)

Built on Flower, this framework provides researchers and developers with tools to configure, execute, and analyze federated learning simulations across diverse datasets and aggregation strategies.

**Core Capabilities:**

- **Byzantine-Tolerant Aggregation**: RFA, Bulyan, Multi-Krum, Trimmed Mean, and PID-based strategies with strategy-specific configurations
- **Attack Simulation**: Configurable attack schedules with label flipping, Gaussian noise, brightness manipulation, and token replacement
- **Dataset Flexibility**: Preconfigured datasets across medical imaging, transportation, and NLP with IID/non-IID distributions (most download automatically on first use)
- **LLM Support**: LoRA-based fine-tuning, masked language modeling, and token-level customization
- **Developer Tools**: Mock data generation, comprehensive testing suite, and interactive demos
- **Cross-Platform**: Automated setup on UNIX/Windows with CPU/GPU/CUDA support

---

## 🚀 Quick Start

### Linux/macOS

```bash
# Setup environment (creates venv and installs dependencies)
sh reinstall_requirements.sh

# Run example simulation
sh run_simulation.sh
```

### Windows (Git Bash or WSL)
>
> 🪟 **Windows Users**: This framework requires **Git Bash** or **WSL** to run setup scripts.  
> [Download Git Bash](https://git-scm.com/downloads) | [WSL Setup Guide](https://learn.microsoft.com/en-us/windows/wsl/install)

```bash
# First install Git Bash: https://git-scm.com/downloads
# Then run in Git Bash:
sh reinstall_requirements.sh
sh run_simulation.sh
```

**Results** are saved to `out/` directory with plots and `.csv` metrics.

> **Manual Setup**: For custom Python environments or troubleshooting, see [Manual Setup](#️-manual-setup--configuration)

---

## 📋 Prerequisites

- Python 3.9, 3.10, or 3.11

## ⚙️ Configuration

**See [Configuration Quick Start](docs/config/quick-start.md) for ready-to-run templates | [Parameter Reference](docs/config/parameters.md) for all configuration options**

### Quick Config Overview

Configs are JSON files with two main sections:

```json
{
  "shared_settings": {
    // Parameters constant across all strategies
    "dataset_keyword": "femnist_iid",
    "num_of_rounds": 20,
    "num_of_clients": 10
    // ... all other parameters
  },
  "simulation_strategies": [
    {
      // Parameters that vary for each strategy
      "aggregation_strategy_keyword": "trust"
    }
  ]
}
```

**Key Parameters**:

- `aggregation_strategy_keyword`: Byzantine-tolerant strategy (`trust`, `multi-krum`, `bulyan`, `rfa`, `trimmed_mean`, `pid` variants)
- `dataset_keyword`: Datasets spanning medical imaging, NLP, and handwriting with IID/non-IID distributions
- `attack_schedule`: Round-based attack scheduling (label flipping, noise, brightness) - see [Attack Scheduling Guide](docs/attack-scheduling.md)
- `num_of_rounds`, `num_of_clients`, `num_of_malicious_clients`: Simulation scale
- `training_device`: Hardware (`auto`, `cpu`, `gpu`, `cuda`) - `auto` automatically detects CUDA with CPU fallback

**Example config**: `config/simulation_strategies/example_strategy_config.json`

### 📋 Configuration Parameter Reference

**Complete parameter registry** - all available configuration options:

#### Common Parameters

| Parameter | Type | Values | Default | Description |
|-----------|------|--------|---------|-------------|
| `aggregation_strategy_keyword` | string | `"trust"`, `"pid"`, `"pid_scaled"`, `"pid_standardized"`, `"pid_standardized_score_based"`, `"multi-krum"`, `"krum"`, `"multi-krum-based"`, `"trimmed_mean"`, `"rfa"`, `"bulyan"`, `"fedavg"` | Required | Byzantine-tolerant aggregation strategy |
| `dataset_keyword` | string | `"femnist_iid"`, `"femnist_niid"`, `"its"`, `"pneumoniamnist"`, `"flair"`, `"bloodmnist"`, `"medquad"`, `"financial_phrasebank"`, `"lexglue"`, `"lung_photos"`, `"breastmnist"`, `"pathmnist"`, `"dermamnist"`, `"octmnist"`, `"retinamnist"`, `"tissuemnist"`, `"organamnist"`, `"organcmnist"`, `"organsmnist"` | Required | Dataset for training |
| `num_of_rounds` | integer | > 0 | Required | Total federated learning rounds |
| `num_of_clients` | integer | > 0 | Required | Number of participating clients |
| `num_of_malicious_clients` | integer | ≥ 0 | Required | Static malicious clients (deprecated - use `attack_schedule`) |
| `strict_mode` | string | `"true"`, `"false"` | `"true"` | Enforce all clients participate every round |
| `remove_clients` | string | `"true"`, `"false"` | Required | Enable Byzantine client removal |
| `show_plots` | string | `"true"`, `"false"` | Required | Display plots during runtime |
| `save_plots` | string | `"true"`, `"false"` | Required | Save plots to `out/` directory |
| `save_csv` | string | `"true"`, `"false"` | Required | Save metrics as CSV files |
| `preserve_dataset` | string | `"true"`, `"false"` | `"false"` | Keep poisoned dataset after execution (incompatible with `attack_schedule`) |
| `training_subset_fraction` | number | 0.0-1.0 | Required | Fraction of data used for training vs evaluation |
| `model_type` | string | `"cnn"`, `"transformer"` | `"cnn"` | Neural network architecture type |
| `training_device` | string | `"auto"`, `"cpu"`, `"gpu"`, `"cuda"` | `"auto"` | Hardware device (`auto` detects CUDA with CPU fallback) |
| `cpus_per_client` | integer | > 0 | Required | CPU cores allocated per client |
| `gpus_per_client` | number | ≥ 0 | Required | GPU fraction per client (0 for CPU-only) |
| `num_of_client_epochs` | integer | > 0 | Required | Local training epochs per round |
| `batch_size` | integer | > 0 | Required | Training batch size |

#### Attack Scheduling (Dynamic Poisoning)

| Parameter | Type | Values | Default | Description |
|-----------|------|--------|---------|-------------|
| `attack_schedule` | array | List of attack entries | `[]` | Round-based attack configuration (see below) |

**Attack Schedule Entry Structure:**

| Field | Type | Values | Required | Description |
|-------|------|--------|----------|-------------|
| `start_round` | integer | 1 to `num_of_rounds` | Yes | First round to apply attack |
| `end_round` | integer | 1 to `num_of_rounds` | Yes | Last round to apply attack |
| `attack_type` | string | `"label_flipping"`, `"gaussian_noise"`, `"brightness"`, `"token_replacement"` | Yes | Type of attack |
| `selection_strategy` | string | `"specific"`, `"random"`, `"percentage"` | Yes | How to select malicious clients |
| `malicious_client_ids` | array | List of client IDs | For `"specific"` | Exact clients to attack |
| `malicious_client_count` | integer | 1 to `num_of_clients` | For `"random"` | Number of random clients |
| `malicious_percentage` | number | 0.0-1.0 | For `"percentage"` | Fraction of clients to attack |

**Attack-Specific Parameters:**

| Attack Type | Parameter | Type | Required | Description |
|-------------|-----------|------|----------|-------------|
| `label_flipping` | `flip_fraction` | number (0.0-1.0) | Yes | Fraction of labels to flip (each sample gets independently random new label unless `target_class` specified) |
| `label_flipping` | `target_class` | integer | No | If specified, all flipped labels assigned to this class (targeted attack); if omitted, each sample gets independent random label (untargeted attack) |
| `gaussian_noise` | `target_noise_snr` | number (dB) | Yes | Signal-to-noise ratio in decibels |
| `gaussian_noise` | `attack_ratio` | number (0.0-1.0) | Yes | Fraction of samples to poison |
| `brightness` | `factor` | number | Yes | Brightness multiplier (0.0=black, 1.0=unchanged, >1.0=brighter) |
| `token_replacement` | `target_vocabulary` | string | Yes | Predefined vocabulary to target (see table below) |
| `token_replacement` | `replacement_strategy` | string | Yes | Replacement strategy to use (see table below) |
| `token_replacement` | `replacement_probability` | number (0.0-1.0) | Yes | Probability of replacing each target token |

**Token Replacement Vocabularies:**

| Vocabulary Name | Description | Example Tokens |
|----------------|-------------|----------------|
| `medical` | Medical/healthcare domain terms | treatment, diagnosis, doctor, hospital, patient, vaccine, surgery, symptom, disease, medication |
| `financial` | Financial/market domain terms | stock, profit, investment, revenue, market, trading, portfolio, asset, bond, equity |
| `legal` | Legal/court domain terms | plaintiff, defendant, court, judge, attorney, lawsuit, trial, verdict, evidence, statute |

**Replacement Strategies:**

| Strategy Name | Description | Example Replacements |
|--------------|-------------|---------------------|
| `negative` | Negative/harmful words for misinformation attacks | avoid, refuse, harmful, dangerous, unsafe, ineffective, pointless, misleading |
| `positive` | Positive/beneficial words for testing defenses | beneficial, effective, helpful, safe, proven, recommended, essential |

> **Note**: All vocabularies and strategies are defined in `src/attack_utils/token_vocabularies.py`. The vocabulary-based system follows 2025 NLP poisoning research best practices for targeted, semantically-aware attacks.

**Example:**

```json
"attack_schedule": [
  {
    "start_round": 3,
    "end_round": 8,
    "attack_type": "label_flipping",
    "flip_fraction": 0.7,
    "selection_strategy": "specific",
    "malicious_client_ids": [0, 1, 2]
  },
  {
    "start_round": 5,
    "end_round": 10,
    "attack_type": "gaussian_noise",
    "target_noise_snr": 10.0,
    "attack_ratio": 1.0,
    "selection_strategy": "percentage",
    "malicious_percentage": 0.2
  }
]
```

> **Note**: Overlapping attacks with different types will stack and apply sequentially. See [Attack Scheduling Guide](docs/attack-scheduling.md) for detailed examples.

#### Strategy-Specific Parameters

**Trust Strategy:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `begin_removing_from_round` | integer | Required | First round to start removing clients |
| `trust_threshold` | number (0-1) | Required | Threshold for client removal |
| `beta_value` | number | Required | Trust calculation constant |

**PID Strategies** (`pid`, `pid_scaled`, `pid_standardized`, `pid_standardized_score_based`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Kp` | number | Required | Proportional gain |
| `Ki` | number | Required | Integral gain |
| `Kd` | number | Required | Derivative gain |
| `num_std_dev` | number | Required | Standard deviations for threshold |

**Krum Strategies** (`krum`, `multi-krum`, `multi-krum-based`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_krum_selections` | integer | Required | Number of clients to select |

**Trimmed Mean Strategy:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trim_ratio` | number (0-0.5) | Required | Fraction of extreme values to discard |

**Bulyan Strategy:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_krum_selections` | integer | Required | Multi-Krum selections for first filtering step |

#### LLM Parameters (for `medquad` dataset)

| Parameter | Type | Values | Default | Description |
|-----------|------|--------|---------|-------------|
| `use_llm` | string | `"true"`, `"false"` | `"false"` | Enable LLM training |
| `llm_model` | string | `"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"`, `"distilbert-base-uncased"` | Required if `use_llm=true` | Pretrained model |
| `llm_task` | string | `"mlm"` | `"mlm"` | Task type (masked language modeling) |
| `llm_chunk_size` | integer | > 0 | Required | Token sequence length |
| `mlm_probability` | number (0-1) | 0.15 | Masking probability for MLM | |
| `llm_finetuning` | string | `"full"`, `"lora"` | `"full"` | Fine-tuning method |
| `lora_rank` | integer | > 0 | For `lora` | Rank of low-rank matrices |
| `lora_alpha` | integer | > 0 | For `lora` | Scaling factor |
| `lora_dropout` | number (0-1) | For `lora` | Dropout rate | |

**See also:**

- [Configuration Quick Start](docs/config/quick-start.md) - Ready-to-run templates
- [Attack Scheduling Guide](docs/attack-scheduling.md) - Detailed attack examples and best practices
- [Parameter Reference](docs/config/parameters.md) - Extended parameter documentation

---

## ⚙️ Manual Setup & Configuration

### Python Environment Setup

**Requirements**: Python 3.9, 3.10, or 3.11 (developed with 3.10.14)

**Manual installation** (Windows or custom environments):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment:

# Git Bash (recommended for Windows)
source venv/Scripts/activate

# Windows CMD/PowerShell (Python commands only - cannot run .sh scripts)
venv\Scripts\activate.bat        # CMD
venv\Scripts\Activate.ps1        # PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Configuration Workflow

1. **Create config file** in `config/simulation_strategies/`
   - Start from `example_strategy_config.json` as template
   - See [Parameter Reference](docs/config/parameters.md) for all parameter options

2. **Run simulation** with CLI arguments:

   ```bash
   # Use default config (example_strategy_config.json)
   python -m src.simulation_runner

   # Specify custom config file
   python -m src.simulation_runner your_config.json

   # Enable debug logging
   python -m src.simulation_runner your_config.json --log-level DEBUG

   # Or use automation script (UNIX/Git Bash)
   sh run_simulation.sh
   ```

   **CLI Options**:
   - `config` - Config filename in `config/simulation_strategies/` (default: `example_strategy_config.json`)
   - `--log-level` - Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

3. **View results** in `out/` directory:
   - Plots (`.png` files if `save_plots: true`)
   - Metrics (`.csv` files if `save_csv: true`)
   - Poisoned datasets (if `preserve_dataset: true`)

### Troubleshooting

**GPU/CUDA setup**: Use `training_device: "auto"` for automatic detection with CPU fallback, or explicitly set `"cuda"` (requires PyTorch/CUDA drivers)

**Dataset downloads**: First run downloads datasets automatically (may take several minutes)

**Config validation errors**: Check [Troubleshooting Guide](docs/config/troubleshooting.md) for common errors and fixes. Use `--log-level DEBUG` for detailed error messages.

---

## 📚 Developer Documentation

### Testing & Development

For detailed testing and development guidelines, see:

- **[Testing Guide](docs/testing-guide.md)** 🧪 - Complete testing documentation including:
  - Quick start commands & development workflow
  - Test development standards & patterns
  - Mock data generation & strategy testing
  - Performance optimization & quality checks

- **[Interactive Demos](tests/demo/README.md)** 🎭 - Hands-on learning through runnable examples:
  - FL mock data generation showcase
  - Test failure analysis & debugging
  - Cross-platform demo launcher
  - Edge case handling examples

### Configuration Guides

- **Configuration Guides** ⚙️ - Complete configuration documentation:
  - [Quick Start](docs/config/quick-start.md) - Ready-to-run templates
  - [Parameters](docs/config/parameters.md) - Complete parameter reference
  - [Strategies](docs/config/strategies.md) - Strategy selection and comparison
  - [Troubleshooting](docs/config/troubleshooting.md) - Common errors and fixes
- **[Attack Scheduling Guide](docs/attack-scheduling.md)** 🎯 - Detailed documentation for configuring attack schedules with round-based scheduling, client selection strategies, and attack stacking

---

## 📊 Description of Collected Metrics

- **Client-Level Metrics**:
  - `Loss`: Provided by Flower.
  - `Accuracy`: Evaluation accuracy (provided by Flower).
  - `Removal criteria`: Used for client removal by the strategy.
  - `Distance`: From cluster center (via KMeans).
  - `Normalized distance`: Scaled distance (MinMaxScaler).

- **Round-Level Metrics**:
  - `Average loss`: Across participating clients.
  - `Average accuracy`: Across participating clients.

---

## 📈 Strategy Comparison

The framework supports executing multiple strategies sequentially and comparing their metrics on a single plot. This enables systematic testing of how different parameters affect training outcomes.

**Quick example** - Compare aggregation strategies:

```json
{
  "shared_settings": {
    "num_of_rounds": 30,
    "num_of_clients": 10
    // ... other constant parameters
  },
  "simulation_strategies": [
    {"aggregation_strategy_keyword": "trust"},
    {"aggregation_strategy_keyword": "multi-krum"},
    {"aggregation_strategy_keyword": "bulyan"}
  ]
}
```

Results are plotted together showing each strategy's loss/accuracy curves for easy comparison.

**See [Strategy Comparison Guide](docs/config/strategies.md#strategy-comparison-examples)** for detailed examples including:

- Varying local epochs to test convergence
- Testing defenses against different numbers of attackers
- Comparing strategy-specific parameter tuning
