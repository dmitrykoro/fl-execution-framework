# Strategy Selection & Configuration Guide

> Understand which aggregation strategy fits your research goals and how to configure it properly.

## Table of Contents

- [Choosing a Strategy](#choosing-a-strategy)
- [Strategy-Specific Configurations](#strategy-specific-configurations)
- [Strategy Comparison Examples](#strategy-comparison-examples)

---

## Choosing a Strategy

### Decision Tree

```text
START: What is your primary goal?
│
├─ Baseline/No attacks
│  └─ Use: "trust" (simplest, good for learning)
│
├─ Defend against label flipping
│  ├─ Small # of attackers (< 1/3)
│  │  └─ Use: "multi-krum" or "krum"
│  └─ Larger # of attackers
│     └─ Use: "bulyan" (most robust)
│
├─ Defend against data poisoning (noise)
│  └─ Use: "rfa" or "trimmed_mean"
│
├─ Research/experimentation
│  └─ Use: PID variants ("pid_standardized", "pid_scaled")
│
└─ Compare strategies
   └─ Use: Multiple strategies in simulation_strategies array
```

### Strategy Summary Table

| Strategy | Best For | Complexity | Client Removal | Key Parameters |
|----------|----------|------------|----------------|----------------|
| `trust` | Learning, baseline | Low | Yes | `trust_threshold`, `beta_value` |
| `krum` | Few attackers, single selection | Medium | Optional | `num_krum_selections` (set to 1) |
| `multi-krum` | Few attackers, multiple selection | Medium | Optional | `num_krum_selections` |
| `multi-krum-based` | Permanent removal needed | Medium | Yes (permanent) | `num_krum_selections` |
| `rfa` | Data poisoning defense | Medium | No | None |
| `trimmed_mean` | Outlier detection | Low | No | `trim_ratio` |
| `bulyan` | Maximum robustness | High | No | None (uses internal Multi-Krum + Trimmed-Mean) |
| `pid` variants | Research, adaptive control | High | Yes | `Kp`, `Ki`, `Kd`, `num_std_dev` |

---

## Strategy-Specific Configurations

### Trust & Reputation (`trust`)

**How it works**: Tracks client trustworthiness over time using distance-based metrics.

**Implementation**: `src/simulation_strategies/trust_based_removal_strategy.py`

**Required parameters**:

```json
{
  "aggregation_strategy_keyword": "trust",
  "trust_threshold": 0.15,
  "beta_value": 0.75,
  "num_of_clusters": 1,
  "begin_removing_from_round": 4,
  "remove_clients": "true"
}
```

**Parameter explanations**:

- `trust_threshold` (0.0-1.0): Lower = more aggressive removal. Start with 0.15
- `beta_value` (0.0-1.0): Weight for historical trust. Higher = more memory. Try 0.75
- `num_of_clusters`: Must be 1 (constraint of current implementation)
- `begin_removing_from_round`: Wait a few rounds for metrics to stabilize (typically 3-5)

**Common values**:

- Conservative: `trust_threshold: 0.1`, `beta_value: 0.8`
- Aggressive: `trust_threshold: 0.2`, `beta_value: 0.5`

---

### Multi-Krum (`multi-krum`, `krum`, `multi-krum-based`)

**How it works**: Selects clients based on closeness to neighbors (detects outliers).

**Implementation**:

- `multi-krum` / `krum`: `src/simulation_strategies/mutli_krum_strategy.py`
- `multi-krum-based`: `src/simulation_strategies/multi_krum_based_removal_strategy.py`

**Required parameters**:

```json
{
  "aggregation_strategy_keyword": "multi-krum",
  "num_krum_selections": 7,
  "remove_clients": "true"
}
```

**Parameter explanations**:

- `num_krum_selections`: How many clients to keep per round
  - Formula: `num_clients - num_malicious_clients - 2` (recommended)
  - For 10 clients, 2 malicious: use 7
  - For `krum` specifically: set to 1

**Variant differences**:

- `krum`: Only uses single best client (`num_krum_selections: 1`)
- `multi-krum`: Removes clients only in current round (temporary)
- `multi-krum-based`: Removes clients permanently across rounds

**Example for different client counts**:

```json
// 20 clients, 4 malicious
"num_of_clients": 20,
"num_of_malicious_clients": 4,
"num_krum_selections": 14  // 20 - 4 - 2

// 10 clients, 3 malicious
"num_of_clients": 10,
"num_of_malicious_clients": 3,
"num_krum_selections": 5  // 10 - 3 - 2
```

---

### RFA (Robust Federated Averaging) (`rfa`)

**How it works**: Uses weighted median aggregation (robust to outliers).

**Implementation**: `src/simulation_strategies/rfa_based_removal_strategy.py`

**Required parameters**:

```json
{
  "aggregation_strategy_keyword": "rfa"
}
```

**Notes**:

- No strategy-specific parameters needed
- Does NOT remove clients
- Set `remove_clients: "false"`
- Best for data poisoning (noise) attacks
- Works well with larger client populations

---

### Trimmed Mean (`trimmed_mean`)

**How it works**: Removes extreme values from both ends before averaging.

**Implementation**: `src/simulation_strategies/trimmed_mean_based_removal_strategy.py`

**Required parameters**:

```json
{
  "aggregation_strategy_keyword": "trimmed_mean",
  "trim_ratio": 0.2
}
```

**Parameter explanations**:

- `trim_ratio` (0.0-0.5): Fraction to trim from each end
  - 0.2 = removes 20% lowest + 20% highest (40% total)
  - 0.3 = removes 30% lowest + 30% highest (60% total)
  - Must be < 0.5 (can't remove everything!)

**Choosing trim_ratio**:

- For `f` malicious clients out of `n` total: use `trim_ratio ≥ f/n`
- Example: 2 malicious / 10 total = 0.2 trim_ratio minimum
- Add buffer: use 0.25-0.3 for safety

---

### Bulyan (`bulyan`)

**How it works**: Two-stage filtering (Multi-Krum → Trimmed-Mean).

**Implementation**: `src/simulation_strategies/bulyan_strategy.py`

**Required parameters**:

```json
{
  "aggregation_strategy_keyword": "bulyan"
}
```

**Notes**:

- No strategy-specific parameters (uses internal defaults)
- Most robust against Byzantine attacks
- Higher computational cost
- Requires sufficient honest clients: `n ≥ 4f + 3` where `f` = malicious
  - For 2 malicious: need 11+ clients
  - For 3 malicious: need 15+ clients

---

### PID Variants (`pid`, `pid_scaled`, `pid_standardized`, `pid_standardized_score_based`)

**How it works**: Control-theory approach using Proportional-Integral-Derivative.

**Implementation**: `src/simulation_strategies/pid_based_removal_strategy.py`

**Required parameters**:

```json
{
  "aggregation_strategy_keyword": "pid_standardized",
  "Kp": 1.0,
  "Ki": 0.1,
  "Kd": 0.05,
  "num_std_dev": 2.0,
  "remove_clients": "true"
}
```

**Parameter explanations**:

- `Kp`: Proportional gain (current error). Start with 1.0
- `Ki`: Integral gain (accumulated error). Start small (0.05-0.1)
- `Kd`: Derivative gain (rate of change). Start very small (0.01-0.05)
- `num_std_dev`: Threshold sensitivity. 2.0 = 95% confidence interval

**Variant differences**:

- `pid`: Original formula
- `pid_scaled`: Integral divided by current round number
- `pid_standardized`: Integral standardized by distribution (recommended)
- `pid_standardized_score_based`: Threshold based on PID scores

**Tuning tips**:

- More aggressive removal: increase `Kp`, decrease `num_std_dev`
- More conservative: decrease `Kp`, increase `num_std_dev`
- If oscillating: reduce `Kd`
- If slow to react: increase `Kp`

---

## Strategy Comparison Examples

### Understanding Config Structure

The framework allows you to execute **multiple aggregation strategies** sequentially and compare their metrics on a single plot. This is powerful for research: you can test how different parameters affect training outcomes.

**Configuration Format**:

```json
{
  "shared_settings": {
    // Parameters that stay CONSTANT across all strategies
  },
  "simulation_strategies": [
    {
      // Parameters that VARY for strategy 1
    },
    {
      // Parameters that VARY for strategy 2
    }
  ]
}
```

**Key Concepts**:

- **`shared_settings`**: Parameters you don't want to change between experiments (e.g., dataset, number of rounds)
- **`simulation_strategies`**: Array of parameter sets that vary for each strategy
- The framework runs each strategy in the array sequentially
- After all strategies complete, it plots metrics (loss, accuracy) on a single graph for comparison

**Important Limitation**:

⚠️ **Cannot vary `num_of_rounds`** - This parameter MUST be in `shared_settings` (all strategies run for the same number of rounds)

---

### Example 1: Testing Local Epoch Count

**Research question**: How does the number of local client epochs affect training metrics?

**Strategy**: Vary `num_of_client_epochs`, keep everything else constant

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "trust",
    "dataset_keyword": "femnist_iid",
    "num_of_rounds": 100,
    "begin_removing_from_round": 4,
    "num_of_clients": 10,
    "num_of_malicious_clients": 2,
    "attack_schedule": [
      {
        "attack_type": "label_flipping",
        "flip_fraction": 0.5,
        "start_round": 1,
        "end_round": 100,
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
    "batch_size": 32,

    "trust_threshold": 0.15,
    "beta_value": 0.75,
    "num_of_clusters": 1
  },
  "simulation_strategies": [
    {
      "num_of_client_epochs": 1
    },
    {
      "num_of_client_epochs": 2
    },
    {
      "num_of_client_epochs": 3
    }
  ]
}
```

**What happens**:

1. Executes 3 strategies total (1 epoch, 2 epochs, 3 epochs)
2. Each strategy runs for 100 rounds with the same dataset/attack
3. After completion, plots show all 3 strategies' loss/accuracy curves on one graph
4. You can visually compare: "Did more epochs help convergence?"

---

### Example 2: Testing Number of Malicious Clients

**Research question**: How does the `trust` strategy perform with different numbers of attackers?

**Strategy**: Vary `num_of_malicious_clients`, keep aggregation strategy constant

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "trust",
    "dataset_keyword": "pneumoniamnist",
    "num_of_rounds": 30,
    "num_of_clients": 20,
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

    "trust_threshold": 0.15,
    "beta_value": 0.75,
    "num_of_clusters": 1,
    "begin_removing_from_round": 4
  },
  "simulation_strategies": [
    {
      "num_of_malicious_clients": 2,
      "attack_schedule": [
        {
          "attack_type": "label_flipping",
          "flip_fraction": 1.0,
          "start_round": 1,
          "end_round": 30,
          "selection_strategy": "percentage",
          "malicious_percentage": 0.1
        }
      ]
    },
    {
      "num_of_malicious_clients": 4,
      "attack_schedule": [
        {
          "attack_type": "label_flipping",
          "flip_fraction": 1.0,
          "start_round": 1,
          "end_round": 30,
          "selection_strategy": "percentage",
          "malicious_percentage": 0.2
        }
      ]
    },
    {
      "num_of_malicious_clients": 6,
      "attack_schedule": [
        {
          "attack_type": "label_flipping",
          "flip_fraction": 1.0,
          "start_round": 1,
          "end_round": 30,
          "selection_strategy": "percentage",
          "malicious_percentage": 0.3
        }
      ]
    }
  ]
}
```

**What happens**:

1. Runs trust strategy 3 times: with 2, 4, then 6 malicious clients
2. All other params identical (same dataset, rounds, trust params)
3. Comparison plot shows: "At what point does trust strategy fail?"

**Note**: When varying `num_of_malicious_clients`, you typically need to also update `attack_schedule` in each strategy entry (since attack schedule depends on number of malicious clients).

---

### Example 3: Comparing Aggregation Strategies

**Research question**: Which performs better: `trust` or `pid`?

**Strategy**: Vary `aggregation_strategy_keyword` (and strategy-specific params)

**Method 1 - Strategy-specific params in each entry:**

```json
{
  "shared_settings": {
    "dataset_keyword": "femnist_iid",
    "num_of_rounds": 30,
    "num_of_clients": 20,
    "num_of_malicious_clients": 2,
    "attack_schedule": [
      {
        "attack_type": "label_flipping",
        "flip_fraction": 0.5,
        "start_round": 1,
        "end_round": 30,
        "selection_strategy": "percentage",
        "malicious_percentage": 0.1
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
    "batch_size": 32
  },
  "simulation_strategies": [
    {
      "aggregation_strategy_keyword": "trust",
      "trust_threshold": 0.15,
      "beta_value": 0.75,
      "num_of_clusters": 1,
      "begin_removing_from_round": 4
    },
    {
      "aggregation_strategy_keyword": "pid",
      "Kp": 1.0,
      "Ki": 0.1,
      "Kd": 0.05,
      "num_std_dev": 2.0
    }
  ]
}
```

**Method 2 - Strategy-specific params in shared_settings (equivalent):**

Since we're not varying the strategy-specific parameters themselves (only the strategy choice), we can put ALL params in `shared_settings`:

```json
{
  "shared_settings": {
    "dataset_keyword": "femnist_iid",
    "num_of_rounds": 30,
    "num_of_clients": 20,
    "num_of_malicious_clients": 2,
    "attack_schedule": [
      {
        "attack_type": "label_flipping",
        "flip_fraction": 0.5,
        "start_round": 1,
        "end_round": 30,
        "selection_strategy": "percentage",
        "malicious_percentage": 0.1
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

    // Trust-specific params (ignored by PID)
    "trust_threshold": 0.15,
    "beta_value": 0.75,
    "num_of_clusters": 1,
    "begin_removing_from_round": 4,

    // PID-specific params (ignored by Trust)
    "Kp": 1.0,
    "Ki": 0.1,
    "Kd": 0.05,
    "num_std_dev": 2.0
  },
  "simulation_strategies": [
    {
      "aggregation_strategy_keyword": "trust"
    },
    {
      "aggregation_strategy_keyword": "pid"
    }
  ]
}
```

**Both methods produce identical results!**

**Why this works**: Each strategy only uses its own parameters. Trust ignores PID params, and vice versa.

**When to use each method**:

- **Method 1**: Clearer, better for documentation/sharing
- **Method 2**: Shorter, better when you have many strategies with complex params

---

### Design Philosophy

The configuration design provides **maximum flexibility**:

1. **Put constant params in `shared_settings`** - Reduces duplication
2. **Put varying params in `simulation_strategies` array** - Makes experiments explicit
3. **Put strategy-specific params wherever convenient** - Both methods work
4. **Vary ANY parameter except `num_of_rounds`** - Test any hypothesis

**Common patterns**:

```json
// Test different aggregation strategies
"simulation_strategies": [
  {"aggregation_strategy_keyword": "trust"},
  {"aggregation_strategy_keyword": "multi-krum"},
  {"aggregation_strategy_keyword": "bulyan"}
]

// Test different attack intensities
"simulation_strategies": [
  {"attack_schedule": [{"flip_fraction": 0.2, ...}]},
  {"attack_schedule": [{"flip_fraction": 0.5, ...}]},
  {"attack_schedule": [{"flip_fraction": 0.8, ...}]}
]

// Test different client counts
"simulation_strategies": [
  {"num_of_clients": 5},
  {"num_of_clients": 10},
  {"num_of_clients": 20}
]

// Test different strategy parameters
"simulation_strategies": [
  {"Kp": 0.5},
  {"Kp": 1.0},
  {"Kp": 2.0}
]
```

---

## 📚 Related Guides

- **Quick Start**: See [Quick Start Templates](quick-start.md) for ready-to-run configs
- **Parameter Reference**: See [Parameter Reference](parameters.md) for detailed parameter documentation
- **Troubleshooting**: See [Troubleshooting Guide](troubleshooting.md) for common errors and fixes
- **Attack Scheduling**: See [Attack Scheduling Guide](../attack-scheduling.md) for round-based poisoning
- **Main README**: Back to [README](../../README.md)
