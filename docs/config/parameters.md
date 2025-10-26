# Configuration Parameter Reference

> Comprehensive reference for all simulation parameters. Use Ctrl+F to quickly find what you need.

**Config Schema**: `src/config_loaders/validate_strategy_config.py`
**Config Model**: `src/data_models/simulation_strategy_config.py`

## Table of Contents

- [Core Simulation Parameters](#core-simulation-parameters)
- [Output & Monitoring Parameters](#output--monitoring-parameters)
- [Training Parameters](#training-parameters)
- [LLM-Specific Parameters](#llm-specific-parameters)

---

## Core Simulation Parameters

### `dataset_keyword`

**What it does**: Selects which dataset to use for training.

**Implementation**: `src/dataset_handlers/dataset_handler.py` - Dataset loading and preparation

**Available options**:

```text
Medical Imaging (all IID unless noted):
- pneumoniamnist - Binary classification
- breastmnist - Binary classification *
- bloodmnist - Unbalanced multi-class classification
- pathmnist - 9-class tissue classification *
- dermamnist - 7-class dermatology classification *
- octmnist - 4-class retinal OCT classification *
- retinamnist - 5-class retinopathy classification *
- tissuemnist - 8-class kidney tissue classification *
- organamnist/organcmnist/organsmnist - 11-class organ classification *
- lung_photos - Non-IID real CT scans
- flair - Binary classification

Handwriting & Vision:
- femnist_iid - Handwritten characters, IID distribution
- femnist_niid - Handwritten characters with letters, non-IID distribution
- its - Non-IID traffic sign classification

NLP:
- medquad - Medical Q&A, requires LLM

* Datasets marked with asterisk download automatically on first use
```

**How to choose**:

- Learning FL: Start with `femnist_iid` (balanced, many clients)
- Medical research: Use `*mnist` datasets
- Non-IID testing: Use `femnist_niid`, `lung_photos`, or `its`
- LLM experiments: Use `medquad`

**Client count constraint**: Must set `num_of_clients ≤ max_available_clients` for dataset

---

### `num_of_rounds`

**What it does**: Total training rounds (server aggregations).

**Typical values**:

- Quick testing: 10-15 rounds
- Normal training: 20-30 rounds
- Convergence testing: 50-100 rounds

**Constraints**:

- Must be in `shared_settings` (cannot vary between strategies)
- Minimum: 1
- Practical maximum: 100-200 (diminishing returns)

**Interaction with other parameters**:

- More `num_of_client_epochs` → fewer rounds needed
- Attack schedules must fit within round count
- `begin_removing_from_round` must be < `num_of_rounds`

---

### `num_of_clients`

**What it does**: Total participating clients (honest + malicious).

**Constraints**:

- Must be ≤ dataset's maximum clients
- Must be ≥ `num_of_malicious_clients`
- For Bulyan: must satisfy `num_of_clients ≥ 4 * num_of_malicious_clients + 3`

**Typical values**:

- Small experiments: 5-10 clients
- Normal simulations: 10-20 clients
- Large-scale: 30+ clients

---

### `num_of_malicious_clients`

**What it does**: How many clients are malicious (launch attacks).

**Constraints**:

- Must be < `num_of_clients`
- For effective defense testing: 10-30% of total clients
- Must have `attack_schedule` defined if > 0

**Examples**:

```json
// Light attack (10%)
"num_of_clients": 10,
"num_of_malicious_clients": 1

// Moderate attack (20%)
"num_of_clients": 10,
"num_of_malicious_clients": 2

// Heavy attack (30%)
"num_of_clients": 10,
"num_of_malicious_clients": 3
```

**Special case**: Can be 0 for baseline (no attack) experiments

---

### `attack_schedule`

**What it does**: Defines when and how attacks occur.

**Implementation**: `src/attack_utils/poisoning.py` - See `should_poison_this_round()` and `apply_poisoning_attack()`

**Format**:

```json
"attack_schedule": [
  {
    "attack_type": "label_flipping",
    "flip_fraction": 1.0,
    "start_round": 1,
    "end_round": 30,
    "selection_strategy": "specific",
    "malicious_client_ids": [0, 1]
  }
]
```

**See**: [Attack Scheduling Guide](../attack-scheduling.md) for comprehensive documentation.

**Quick reference**:

| Attack Type | Required Params | Common Values |
|-------------|----------------|---------------|
| `label_flipping` | `flip_fraction` | 0.5 (50%), 1.0 (100%) |
| | `target_class` (optional) | 0, 1, 2, etc. |
| `gaussian_noise` | `target_noise_snr` | 5.0-20.0 dB |
| | `attack_ratio` | 0.3-0.5 |
| `brightness` | `factor` | 0.3 (darker), 1.8 (brighter) |
| `token_replacement` | `replacement_prob` | 0.2-0.5 |
| | `vocab_size` | 30522 (BERT default) |

**Important**:

- Empty array `[]` for no attacks
- `preserve_dataset: "true"` to inspect poisoned data
- When using dynamic attack schedules, `preserve_dataset` is automatically set to `false`

---

### `strict_mode`

**What it does**: Ensures all clients participate in every round.

**Values**: `"true"` or `"false"`

**When enabled** (`"true"`, default):

- Automatically sets:
  - `min_fit_clients = num_of_clients`
  - `min_evaluate_clients = num_of_clients`
  - `min_available_clients = num_of_clients`
- Simulation fails if any client unavailable
- More predictable, reproducible results

**When disabled** (`"false"`):

- Must manually set `min_fit_clients`, `min_evaluate_clients`, `min_available_clients`
- Allows partial client participation
- More realistic for production scenarios

**Recommendation**: Keep as `"true"` for research/learning

---

### `remove_clients`

**What it does**: Controls whether strategy can permanently exclude clients.

**Values**: `"true"` or `"false"`

**Interaction with strategies**:

| Strategy | Requires `remove_clients: "true"` |
|----------|-----------------------------------|
| `trust` | Yes (core functionality) |
| `multi-krum-based` | Yes (permanent removal) |
| `pid` variants | Yes (core functionality) |
| `multi-krum` | Optional (temporary per-round) |
| `krum` | Optional (temporary per-round) |
| `rfa` | No (doesn't remove) |
| `trimmed_mean` | No (doesn't remove) |
| `bulyan` | No (doesn't remove) |

**When to use**:

- `"true"`: Testing client removal mechanisms
- `"false"`: Testing aggregation robustness without removal

---

## Output & Monitoring Parameters

### `show_plots`, `save_plots`, `save_csv`

**What they do**:

- `show_plots`: Display plots during runtime (blocks execution)
- `save_plots`: Save plots to `out/` directory
- `save_csv`: Save metrics as CSV files

**Recommended combinations**:

```json
// Development (interactive)
"show_plots": "true",
"save_plots": "true",
"save_csv": "true"

// Automated runs (no blocking)
"show_plots": "false",
"save_plots": "true",
"save_csv": "true"

// Quick testing (minimal I/O)
"show_plots": "false",
"save_plots": "false",
"save_csv": "false"
```

---

### `preserve_dataset`

**What it does**: Saves poisoned dataset copies to disk for inspection.

**Values**: `"true"` or `"false"`

**When to use**:

- `"true"`: Debugging attacks, verifying poison worked
- `"false"`: Normal runs (saves disk space)

**Important**:

- Large datasets = significant disk usage
- Automatically set to `"false"` for dynamic attack schedules (required)
- Location: Saved to temporary directories during execution

**Constraint**:

- **MUST be `"false"`** when using dynamic attack schedules (varies by round/client)
- Static attack schedules: can be either `"true"` or `"false"`

---

## Training Parameters

### `training_subset_fraction`

**What it does**: Splits each client's data into train/test sets.

**Values**: 0.0-1.0 (typically 0.8-0.95)

**Examples**:

- `0.9`: 90% training, 10% evaluation
- `0.8`: 80% training, 20% evaluation
- `1.0`: All data for training (no evaluation per client)

**Recommendation**: Use 0.9 as standard

---

### `num_of_client_epochs`

**What it does**: Local training epochs per client per round.

**Typical values**:

- Fast iteration: 1 epoch
- Better convergence: 2-3 epochs
- Risk of overfitting: 5+ epochs

**Trade-offs**:

- More epochs → better local fit, longer runtime
- Fewer epochs → faster, more rounds needed

---

### `batch_size`

**What it does**: Samples per training batch.

**Typical values**:

- Small datasets: 8-16
- Medium datasets: 32-64
- Large datasets: 128-256

**Device constraints**:

- GPU: Can handle larger batches (32-256)
- CPU: Use smaller batches (8-32)
- LLMs: Use smaller batches (4-16)

---

### `training_device`

**What it does**: Hardware for training.

**Options**: `"cpu"`, `"gpu"`, `"cuda"`

**How to choose**:

```json
// CPU only (slowest, most compatible)
"training_device": "cpu",
"cpus_per_client": 2,
"gpus_per_client": 0

// CUDA/GPU (fastest, requires NVIDIA GPU)
"training_device": "cuda",
"cpus_per_client": 2,
"gpus_per_client": 0.1  // Share GPU across clients
```

**Requirements**:

- `"cuda"`: Requires PyTorch with CUDA, NVIDIA GPU drivers
- `"gpu"`: Requires compatible GPU (NVIDIA/AMD)
- `"cpu"`: Always works (no special setup)

---

### `model_type`

**What it does**: Specifies the high-level model category to use for training.

**Accepted values**:

- `"cnn"`: Convolutional neural network for vision tasks (MNIST, FEMNIST, ITS, medical imaging)
- `"transformer"`: Transformer-based model for NLP tasks (MedQuAD with LLMs)

**How to choose**:

```json
// Vision tasks (images)
"dataset_keyword": "femnist_iid",
"model_type": "cnn"

// LLM tasks (text)
"dataset_keyword": "medquad",
"use_llm": "true",
"model_type": "transformer",
"llm_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
```

**Notes**:

- `"cnn"`: Configures training loop to use `CrossEntropyLoss` and `Adam` optimizer
- `"transformer"`: Configures training loop to use model-internal loss and `AdamW` optimizer
- Required when `use_llm: "true"`

**Important**: `model_type` controls the *training mode*, not the architecture. The actual CNN architecture is determined by `dataset_keyword`:

- `femnist_iid` → `FemnistReducedIIDNetwork`
- `femnist_niid` → `FemnistFullNIIDNetwork`
- `pneumoniamnist` → `PneumoniamnistNetwork`
- `lung_photos` → `LungCancerCNN`
- Each dataset uses its own optimized architecture

Model type doesn't affect aggregation strategy testing

---

## LLM-Specific Parameters

Only needed when `use_llm: "true"` and `dataset_keyword: "medquad"`.

```json
{
  "use_llm": "true",
  "llm_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
  "llm_finetuning": "lora",
  "llm_task": "mlm",
  "llm_chunk_size": 128,
  "mlm_probability": 0.15,
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "lora_target_modules": ["query", "value"]
}
```

**Parameter explanations**:

- `llm_model`: HuggingFace model name (currently only `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` is supported)
- `llm_finetuning`: `"full"` (all params) or `"lora"` (efficient)
- `llm_task`: Currently only `"mlm"` (masked language modeling)
- `llm_chunk_size`: Token sequence length (128-512, lower for CPU)
- `mlm_probability`: Fraction of tokens to mask (0.15 standard)
- `lora_rank`: Low-rank dimension (4-16, lower = fewer params)
- `lora_alpha`: Scaling factor (typically 2× rank)
- `lora_dropout`: Regularization (0.1 standard)
- `lora_target_modules`: Which layers to adapt (`["query", "value"]` for attention)

**Typical LoRA settings**:

- Fast/lightweight: `rank=4, alpha=8`
- Standard: `rank=8, alpha=16`
- High-quality: `rank=16, alpha=32`

---

## 📚 Related Guides

- **Quick Start**: See [Quick Start Templates](quick-start.md) for ready-to-run configs
- **Strategy Selection**: See [Strategy Guide](strategies.md) for choosing aggregation strategies
- **Troubleshooting**: See [Troubleshooting Guide](troubleshooting.md) for common errors and fixes
- **Attack Scheduling**: See [Attack Scheduling Guide](../attack-scheduling.md) for round-based poisoning
- **Main README**: Back to [README](../../README.md)
