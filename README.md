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
