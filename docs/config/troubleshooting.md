# Configuration Troubleshooting Guide

> Find your error message below and get the fix immediately. Use Ctrl+F to search for your specific error.

**Config Validation**: `src/config_loaders/validate_strategy_config.py` - All validation rules and error messages

## Table of Contents

- [Config Validation Errors](#config-validation-errors)
- [Runtime Errors](#runtime-errors)
- [Common Mistakes](#common-mistakes)
- [Experimentation Tips](#tips-for-experimentation)

---

## Config Validation Errors

### 1. Config Won't Load / Validation Errors

**Error**: `Config validation failed` or `Missing required parameter`

**Checklist**:

- [ ] JSON syntax valid? (Use [jsonlint.com](https://jsonlint.com) to check)
- [ ] Strategy-specific params included? (e.g., `num_krum_selections` for Multi-Krum)
- [ ] `attack_schedule` is a list `[]` not a dict `{}`?
- [ ] All string values in quotes? (`"true"` not `true`)
- [ ] `num_of_malicious_clients > 0` but `attack_schedule` is empty?

**Common fixes**:

```json
// WRONG - boolean without quotes
"strict_mode": true

// RIGHT
"strict_mode": "true"
```

---

### 2. Client Count Mismatch

**Error**: `Not enough clients available` or `Client count exceeds dataset maximum`

**Causes**:

- `num_of_clients` > dataset's max clients
- `num_of_malicious_clients` ≥ `num_of_clients`

**Fix**:

```json
// Check dataset documentation for max available clients
// Example: femnist_iid supports many clients
"dataset_keyword": "femnist_iid",
"num_of_clients": 10,  // ✓ OK
"num_of_malicious_clients": 2  // ✓ OK (2 < 10)
```

---

### 3. Bulyan Fails with "Not Enough Clients"

**Error**: Simulation crashes during Bulyan aggregation

**Cause**: Bulyan requires `n ≥ 4f + 3` where `n` = total clients, `f` = malicious

**Fix**:

```json
// WRONG - 10 clients, 3 malicious
// Requires: 4*3 + 3 = 15 clients
"num_of_clients": 10,
"num_of_malicious_clients": 3

// RIGHT
"num_of_clients": 15,
"num_of_malicious_clients": 3
```

**Quick reference**:

- 1 malicious → need 7+ clients
- 2 malicious → need 11+ clients
- 3 malicious → need 15+ clients
- 4 malicious → need 19+ clients

---

### 4. Multi-Krum Selection Too High

**Error**: `num_krum_selections exceeds available clients`

**Cause**: `num_krum_selections` set too high

**Fix**:

```json
// Formula: num_clients - num_malicious - 2
"num_of_clients": 10,
"num_of_malicious_clients": 2,
"num_krum_selections": 7  // 10 - 2 - 2 = 6 (max), use 5-6
```

---

### 10. Missing Strategy-Specific Parameters

**Error**: `Missing required parameter for [strategy]`

**Quick reference**:

```json
// Trust strategy
"aggregation_strategy_keyword": "trust",
"trust_threshold": 0.15,
"beta_value": 0.75,
"num_of_clusters": 1,
"begin_removing_from_round": 4

// Multi-Krum variants
"aggregation_strategy_keyword": "multi-krum",
"num_krum_selections": 7

// Trimmed Mean
"aggregation_strategy_keyword": "trimmed_mean",
"trim_ratio": 0.2

// PID variants
"aggregation_strategy_keyword": "pid_standardized",
"Kp": 1.0,
"Ki": 0.1,
"Kd": 0.05,
"num_std_dev": 2.0
```

---

## Runtime Errors

### 5. Attack Schedule Not Working

**Symptoms**: No attack visible in metrics, accuracy stays high

**Checklist**:

- [ ] `num_of_malicious_clients > 0`?
- [ ] `start_round` and `end_round` within `num_of_rounds`?
- [ ] `selection_strategy` specified? (e.g., `"percentage"`, `"specific"`, `"random"`)
- [ ] Required attack params included? (e.g., `flip_fraction` for label flipping)

**Debug**:

```json
// Enable dataset preservation to verify attack applied
"preserve_dataset": "true",

// Check attack is within round range
"num_of_rounds": 30,
"attack_schedule": [
  {
    "start_round": 1,  // ✓ OK (1 ≤ 30)
    "end_round": 30    // ✓ OK (≤ num_of_rounds)
  }
]
```

---

### 6. Dynamic Attack Schedule with preserve_dataset

**Error**: `preserve_dataset must be false for dynamic attack schedules`

**Cause**: Config validation enforces this for dynamic attacks (round-varying or client-varying)

**Fix**:

```json
// If using round-based or client-specific attacks
"attack_schedule": [
  {
    "attack_type": "label_flipping",
    "flip_fraction": 1.0,
    "start_round": 5,  // Changes by round
    "end_round": 10,
    "selection_strategy": "specific",
    "malicious_client_ids": [0, 1]  // Client-specific
  }
],
"preserve_dataset": "false"  // MUST be false
```

---

### 7. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Fixes**:

```json
// 1. Reduce batch size
"batch_size": 16,  // Down from 32

// 2. Reduce GPU allocation per client
"gpus_per_client": 0.05,  // Down from 0.1

// 3. Use CPU instead
"training_device": "cpu"
```

---

### 9. LLM Training Fails

**Common errors**: Model not found, token errors, memory issues

**Fixes**:

```json
// 1. Verify model name (must be exact HuggingFace name)
"llm_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",  // ✓ Correct
// NOT "bert-base-uncased" or "bert-base" or "BERT"

// 2. Use medquad dataset only
"dataset_keyword": "medquad",
"use_llm": "true"

// 3. Reduce memory usage
"llm_chunk_size": 64,  // Down from 128
"batch_size": 4,       // Down from 8
"lora_rank": 4         // Down from 8
```

---

## Common Mistakes

### 8. Strategy Comparison Shows Identical Lines

**Symptom**: All strategies produce same plot line

**Cause**: Only varying strategy name, not actual behavior

**Fix**:

```json
// WRONG - PID strategies need different params to differ
"simulation_strategies": [
  {
    "aggregation_strategy_keyword": "pid_standardized",
    "Kp": 1.0,
    "Ki": 0.1,
    "Kd": 0.05
  },
  {
    "aggregation_strategy_keyword": "pid_scaled",
    "Kp": 1.0,  // Same params = same behavior!
    "Ki": 0.1,
    "Kd": 0.05
  }
]

// RIGHT - Vary parameters being tested
"simulation_strategies": [
  {
    "aggregation_strategy_keyword": "pid_standardized",
    "Kp": 1.0
  },
  {
    "aggregation_strategy_keyword": "pid_standardized",
    "Kp": 2.0  // Different Kp to test effect
  }
]
```

---

## Tips for Experimentation

### Start Small, Scale Up

```json
// 1. Quick validation (2-3 minutes)
"num_of_rounds": 5,
"num_of_clients": 5,
"num_of_client_epochs": 1,
"batch_size": 32

// 2. Full experiment (20-30 minutes)
"num_of_rounds": 30,
"num_of_clients": 10,
"num_of_client_epochs": 2,
"batch_size": 32
```

---

### Isolate Variables

When testing, vary ONE thing at a time:

```json
// Test effect of num_of_client_epochs
{
  "shared_settings": {
    "num_of_rounds": 20,  // Fixed
    "num_of_clients": 10,  // Fixed
    "batch_size": 32  // Fixed
  },
  "simulation_strategies": [
    {"num_of_client_epochs": 1},
    {"num_of_client_epochs": 2},
    {"num_of_client_epochs": 3}
  ]
}
```

---

### Save Everything for Important Runs

```json
"show_plots": "true",
"save_plots": "true",
"save_csv": "true",
"preserve_dataset": "true"  // Only if not using dynamic attacks
```

---

### Use Descriptive Config Names

Instead of `config1.json`, use:

- `baseline_no_attack_femnist.json`
- `krum_vs_bulyan_label_flip.json`
- `pid_tuning_kp_values.json`

---

## 📚 Related Guides

- **Quick Start**: See [Quick Start Templates](quick-start.md) for ready-to-run configs
- **Parameter Reference**: See [Parameter Reference](parameters.md) for detailed parameter documentation
- **Strategy Selection**: See [Strategy Guide](strategies.md) for choosing aggregation strategies
- **Attack Scheduling**: See [Attack Scheduling Guide](../attack-scheduling.md) for round-based poisoning
- **Main README**: Back to [README](../../README.md)
