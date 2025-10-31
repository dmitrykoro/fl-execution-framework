# Federated Learning Simulation Framework

> *Framework for Federated Learning benchmarking and deployment configuration investigation.*

[![codecov](https://codecov.io/github/dmitrykoro/fl-execution-framework/graph/badge.svg?token=HJFASRJ43T)](https://codecov.io/github/dmitrykoro/fl-execution-framework)

The framework provides researchers and developers with tools to configure, execute, and analyze Federated Learning
simulations across diverse datasets and aggregation strategies, with the possibility to validate Federated Learning
under real-world data imperfections and adversarial attacks.

**Core Capabilities:**

- **Benchmarking of Byzantine-Tolerant Aggregation**: RFA, Bulyan, Multi-Krum, Trimmed Mean, and PID-based strategies
  benchmarking with client exclusion performance metrics
- **Adversarial Attack and Data Quality Variation Simulation**: Configurable data poisoning attack schedules with label
  flipping, Gaussian noise, brightness manipulation, and token replacement
- **Dataset Flexibility**: Preconfigured datasets across medical imaging, transportation, and NLP with IID/non-IID
  distributions
- **LLM Support**: LoRA-based fine-tuning, masked language modeling, and token-level customization
- **Developer Tools**: Mock data generation, comprehensive testing suite, and interactive demos
- **Cross-Platform**: Automated setup on UNIX/Windows with CPU/GPU/CUDA support

---

## Quick Start

### Linux/macOS

```bash
# Setup environment (creates venv and installs dependencies)
sh reinstall_requirements.sh

# Run example simulation
sh run_simulation.sh
```

### Windows (Git Bash or WSL)

>
> **Windows Users**: This framework requires **Git Bash** or **WSL** to run setup scripts.  
> [Download Git Bash](https://git-scm.com/downloads) | [WSL Setup Guide](https://learn.microsoft.com/en-us/windows/wsl/install)

```bash
# First install Git Bash: https://git-scm.com/downloads
# Then run in Git Bash:
sh reinstall_requirements.sh
sh run_simulation.sh
```

**Results** are saved to `out/` directory with plots and `.csv` metrics.

> **Manual Setup**: For custom Python environments or troubleshooting,
> see [Manual Setup](#️-manual-setup--configuration)

---

## Prerequisites

- Python 3.9, 3.10, or 3.11

## Configuration

**See [Configuration Quick Start](docs/config/quick-start.md) for examples of ready-to-run templates**

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

**Default config**: `config/simulation_strategies/example_strategy_config.json`

### Configuration Parameter Reference

This section includes all possible configuration entries.

#### Common parameters (applicable to all strategies)

- **`aggregation_strategy_keyword`**  
  Defines the aggregation strategy. Options:
    - `trust`: Trust & Reputation-based aggregation.
    - `pid`: PID-based aggregation. Initial version of the formula.
    - `pid_scaled`: PID-based aggregation with Integral part divided by the number of current round, threshold is
      calculated based on client distances.
    - `pid_standardized`: PID-based aggregation with the Integral part standardized based on the distribution parameters
      of all Integral parts, threshold is calculated based on client distances.
    - `pid_standardized_score_based`: Same as pid_standardized, but threshold is calculated based on pid scores.
    - `multi-krum`: Multi-Krum aggregation. Clients are removed from aggregation only in current round.
    - `krum`: Krum aggregation works like Multi-Krum, but uses only a single client.
    - `multi-krum-based`: Multi-Krum-based aggregation where removed clients are excluded from aggregation permanently.
    - `rfa`: RFA (Robust Federated Averaging) aggregation strategy. Provides Byzantine fault tolerance through weighted
      median-based aggregation.
    - `trimmed_mean`: Trimmed-Mean aggregation strategy. Aggregates updates by removing a fixed fraction of the largest
      and smallest values for each parameter dimension before averaging. Robust against outliers and certain types of
      attacks.
    - `bulyan`: Bulyan aggregation strategy. Uses Multi-Krum as the first step of filtering and Trimmed-Mean as the
      second step to ensure robustness.


- **`strict_mode`**: ensures that Flower trains and aggregates all available clients at every round. When enabled (
  default), automatically sets `min_fit_clients`, `min_evaluate_clients`, and `min_available_clients` to equal
  `num_of_clients`. Options: `"true"`, `"false"`.

- **`remove_clients`**: attempt to remove malicious clients using strategy-specific mechanisms.


- **`dataset_keyword`**  
  Dataset used for execution. Options:
    - `femnist_iid`: handwritten digit subset (0-9), 10 classes, IID distribution, 100 clients.
    - `femnist_niid`: same, but the data is distributed in non-iid manner, according to authors' description. 16 clients
      max.
    - `its`: intelligent Transportation Systems domain, binary classification (traffic sign vs stop sign), 12 clients.
    - `pneumoniamnist`: medical imaging (pneumonia diagnosis), binary classification, IID distribution, 10 clients.
    - `flair`: non-IID distribution (FLAIR dataset, unsupported in current version), 20 clients.
    - `bloodmnist`: IID distribution, but non-equal number of samples per class, 40 clients.
    - `lung_photos`: contains images of lung cancer from NLST archive from different CT machines. Data distributed
      according to the source, with varying number of images representing each stage of cancer. 30 clients.
    - `breastmnist`: breast ultrasound images for tumor detection, binary classification (malignant vs benign), 10
      clients.
    - `pathmnist`: histopathologic images of colon tissue, 9 classes, IID distribution, 40 clients.
    - `dermamnist`: dermatological lesion images, 7 classes (various skin diseases), 10 clients.
    - `octmnist`: optical coherence tomography images of retinal tissue, 4 classes, 40 clients.
    - `retinamnist`: retina fundus images for diabetic retinopathy classification, 5 classes, 40 clients.
    - `tissuemnist`: gray-scale microscopic images of human tissue, 8 classes, IID distribution, 40 clients.
    - `organamnist`: axial view CT scans of abdominal organs, 11 classes, 40 clients.
    - `organcmnist`: coronal view CT scans of abdominal organs, 11 classes, 40 clients.
    - `organsmnist`: sagittal view CT scans of abdominal organs, 11 classes, 40 clients.

- `num_of_rounds`: total aggregation rounds.
- `num_of_clients`: number of clients (limited to available dataset clients).
- `num_of_malicious_clients`: number of malicious clients (malicious throughout simulation).
- `attack_type`: type of adversarial attack:
    - `label_flipping`: flip 100% of client labels;
    - `gaussian_noise`: add gaussian noise to client image samples in each label. The following params need to be
      specified:
        - `gaussian_noise_mean`: The mean (μ) of the Gaussian distribution. It’s the average value of the noise, 0 for
          the center. Setting mean > 0 will make the image brighter on average, darker otherwise.
        - `gaussian_noise_std`: (0 - 100). The standard deviation (σ) of the Gaussian distribution, which controls how
          spread out the noise values are. 0 = no noise, 50+ = heavy noise.
        - `attack_ratio`: proportion of samples for each label to poison.

- `show_plots`: show plots during runtime (`true`/`false`).
- `save_plots`: save plots to `out/` directory (`true`/`false`).
- `save_csv`: Save metrics as `.csv` files in `out/` directory (`true`/`false`).
- `preserve_dataset`: save poisoned dataset for verification (`true`/`false`).
- `training_subset_fraction`: fraction of each client's dataset for training (e.g., `0.9` for 90% training, 10%
  evaluation).
- `model_type`: type of model being trained

- `save_attack_snapshots`: save or not attack snapshots (`true`/`false`).
- `snapshot_max_samples`: number of samples per snapshot (1-50, default: 5). 
- `attack_snapshot_format`: 
  - `"pickle"`: serialize attack
  - `"visual"`: visual representation
  - `"pickle_and_visual"`: save both

- **`attack_schedule`** list of attack entries `[]`
    - `start_round`: when the attack should start (1 to `num_of_rounds`).
    - `end_round`: when the attack should end (1 to `num_of_rounds`).
    - `attack_type`:
        - `"label_flipping"`:
            - `flip_fraction`: fraction of samples to flip labels
            - `target_class`: target class for targeted attacks
        - `"gaussian_noise"`:
            - `target_noise_snr`: (dB) resulting signal-to-noise ratio
            - `attack_ratio`: fraction of samples to increase noise
        - `"brightness"`:
            - `factor`: brightness multiplier (0.0=black, 1.0=unchanged, >1.0=brighter
        - `"token_replacement"`:
            - `replacement_prob`: probability of replacing each token (0.0-1.0)
            - `vocab_size`: vocabulary size (default: 30522)
    - `selection_strategy`: how to select malicious clients:
        - `"specific"`:
            - `malicious_client_ids`: exact attacker clients (e.g., `[1, 2]`)
        - `"random"`:
            - `malicious_client_count`: number of randomly selected malicious clients
        - `"percentage"`:
            - `malicious_percentage`: fraction of attacker clients (e.g., 0.2)

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

> **Note**: Overlapping attacks with different types will stack and apply sequentially.
> See [Attack Scheduling Guide](docs/attack-scheduling.md) for detailed examples.

- **Flower settings**:
    - `training_device`: `cpu`, `gpu`, or `cuda`.
    - `cpus_per_client`: processors per client.
    - `gpus_per_client`: GPUs per client (if `cuda` is set as the `training_device`).
    - `min_fit_clients`, `min_evaluate_clients`, `min_available_clients`: client quotas for each round.
    - `evaluate_metrics_aggregation_fn`: not used.
    - `num_of_client_epochs`: local client training epochs per round.
    - `batch_size`: batch size for training.

- **LLM settings**:
    - `use_llm`: use an llm (`true`/`false`)
    - `llm_model`: the llm model to be used
    - `llm_finetuning`: how to finetune the llm (`full`, `lora`)
    - `llm_task`: the task the llm is performing (`mlm`)
    - `llm_chunk_size`: size of the token sequences used for training/testing
    - **MLM settings**
        - `mlm_probability`: specific to mlm tasks, the probability that a token is masked
    - **Lora settings**
        - `lora_rank`: rank/size of the low-rank matrices used in lora
        - `lora_alpha`: scaling factor for lora updates
        - `lora_dropout`: dropout rate applied during training
        - `lora_target_modules`: list of model layers where lora should be applied

#### Strategy-specific parameters

**For `trust` strategy**:

- `begin_removing_from_round`: start round for removing malicious clients.
- `trust_threshold`: threshold for client removal (typically, in the range `0-1`).
- `beta_value`: constant for Trust & Reputation calculus.
- `num_of_clusters`: number of clusters (must be `1`).

**For `pid`, `pid_standardized`, `pid_scaled`, `pid_standardized_score_based` strategies**:

- `num_std_dev`: number of standard deviations used int he calculation of PiD threshold at each round.
- `Kp`, `Ki`, `Kd`: PID controller parameters.

**For `krum`, `multi-krum`, `multi-krum-based` strategies**:

- `num_krum_selections`: how many clients the algorithm will select.

**For `trimmed_mean` strategy**:

- `trim_ratio`: fraction of extreme values to discard from both ends (lowest and highest) of each parameter dimension
  before averaging. Must be in the range 0–0.5.

---

## Manual Setup & Configuration

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

**GPU/CUDA setup**: Use `training_device: "auto"` for automatic detection with CPU fallback, or explicitly set`"cuda"` (
requires PyTorch/CUDA drivers)

**Dataset downloads**: First run downloads datasets automatically (may take several minutes)

**Config validation errors**: Check [Troubleshooting Guide](docs/config/troubleshooting.md) for common errors and fixes.
Use `--log-level DEBUG` for detailed error messages.

---

## Developer Documentation

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
- **[Attack Scheduling Guide](docs/attack-scheduling.md)** 🎯 - Detailed documentation for configuring attack schedules
  with round-based scheduling, client selection strategies, and attack stacking

---

## Description of Collected Metrics

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

## Strategy Comparison

The framework supports executing multiple strategies sequentially and comparing their metrics on a single plot. This
enables systematic testing of how different parameters affect training outcomes.

**Quick example** - Compare aggregation strategies:

```json
{
  "shared_settings": {
    "num_of_rounds": 30,
    "num_of_clients": 10
    // ... other constant parameters
  },
  "simulation_strategies": [
    {
      "aggregation_strategy_keyword": "trust"
    },
    {
      "aggregation_strategy_keyword": "multi-krum"
    },
    {
      "aggregation_strategy_keyword": "bulyan"
    }
  ]
}
```

Results are plotted together showing each strategy's loss/accuracy curves for easy comparison.

**See [Strategy Comparison Guide](docs/config/strategies.md#strategy-comparison-examples)** for detailed examples
including:

- Varying local epochs to test convergence
- Testing defenses against different numbers of attackers
- Comparing strategy-specific parameter tuning
