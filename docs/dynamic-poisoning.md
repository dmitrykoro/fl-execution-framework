# ⚔️ Dynamic Poisoning Attacks

**Round-based attack scheduling for federated learning simulations.**

---

## ⚡ Quick Start

### Configuration

Add `dynamic_attacks` to your strategy config:

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "krum",
    "num_of_rounds": 12,
    "num_of_clients": 10,
    "dataset_keyword": "femnist_iid"
  },
  "simulation_strategies": [
    {
      "attack_type": "label_flipping",
      "dynamic_attacks": {
        "enabled": true,
        "schedule": [
          {
            "start_round": 5,
            "end_round": 10,
            "selection_strategy": "specific",
            "client_ids": [0, 1],
            "attack_config": {
              "type": "label_flipping",
              "params": { "flip_fraction": 0.5, "num_classes": 62 }
            }
          }
        ]
      }
    }
  ]
}
```

### Running

```bash
./run_simulation.sh
```

---

## 🎯 Overview

Control when and which clients are poisoned during training:

- **Round-based scheduling** - Attacks trigger during specific rounds
- **Client targeting** - Specific clients, random selection, or percentage
- **Multi-phase attacks** - Different attack types at different rounds
- **Clean transitions** - Clients train normally → attack → resume training

**Example:**

```text
Rounds 1-4:   Normal training ✅
Rounds 5-10:  Clients 0,1 flip 50% labels ⚠️
Rounds 11-20: Normal training ✅
```

---

## ⚙️ How Poisoning Works

### When Does Poisoning Happen?

Poisoning occurs **during training rounds at the batch level**, not before:

```text
Round 5 (poisoning enabled for client 0)
  ├─ Load batch 1 (32 images) → Check schedule → Poison batch → Train
  ├─ Load batch 2 (32 images) → Check schedule → Poison batch → Train
  ├─ Load batch 3 (32 images) → Check schedule → Poison batch → Train
  └─ ... continue for all batches
```

### Poisoning Flow

For each batch in a poisoned round:

1. **Load batch** - DataLoader provides next batch (e.g., 32 images + labels)
2. **Check schedule** - `should_poison_this_round(current_round, client_id, schedule)`
3. **Apply attack** - If conditions met, poison the batch in-memory
4. **Train** - Model trains on the (possibly poisoned) batch
5. **Repeat** - Next batch

**Key Points:**

- Original dataset is **never modified** - poisoning happens temporarily in memory
- Each batch is checked and poisoned independently
- Poisoning is **real-time during training**, not a preprocessing step
- Once the round ends, poisoned data is discarded (not persistent)

### Code Reference

- `src/attack_utils/poisoning.py:103-149` - Schedule checking logic
- `src/client_models/flower_client.py:67-72` - Batch-level poisoning for CNNs
- `src/client_models/flower_client.py:99-109` - Batch-level poisoning for transformers

---

## 🕒 Schedule Configuration

**Single phase:**

```json
{
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 1,
        "end_round": 10,
        "selection_strategy": "specific",
        "client_ids": [0, 1, 2],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.7, "num_classes": 62 }
        }
      }
    ]
  }
}
```

**Multiple phases:**

```json
{
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 1,
        "end_round": 5,
        "selection_strategy": "specific",
        "client_ids": [0],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.8, "num_classes": 62 }
        }
      },
      {
        "start_round": 10,
        "end_round": 15,
        "selection_strategy": "specific",
        "client_ids": [1, 2],
        "attack_config": {
          "type": "gaussian_noise",
          "params": { "mean": 0.0, "std": 0.2 }
        }
      }
    ]
  }
}
```

---

## 🎯 Client Selection

### Specific Clients

```json
{
  "selection_strategy": "specific",
  "client_ids": [0, 1, 3, 7]
}
```

Use for: Studying specific client behavior in heterogeneous settings

### Random Selection

```json
{
  "selection_strategy": "random",
  "num_clients": 3,
  "_selected_clients": [2, 5, 7]
}
```

Use for: Simulating unpredictable adversaries

> **Note:** `_selected_clients` is set by the framework at runtime

### Percentage-based

```json
{
  "selection_strategy": "percentage",
  "percentage": 0.2,
  "_selected_clients": [1, 4, 6, 9]
}
```

Use for: Scaling attacks with total client count

> **Note:** `_selected_clients` is set by the framework at runtime

---

## 💣 Attack Types

### Label Flipping (Classification)

**Random:**

```json
{
  "type": "label_flipping",
  "params": {
    "flip_fraction": 0.5,
    "num_classes": 62
  }
}
```

**Targeted:**

```json
{
  "type": "label_flipping",
  "params": {
    "flip_fraction": 1.0,
    "num_classes": 62,
    "target_class": 7
  }
}
```

Use for: Testing label-based Byzantine defenses, studying targeted vs untargeted attack detection

---

### Gaussian Noise (Images)

```json
{
  "type": "gaussian_noise",
  "params": {
    "mean": 0.0,
    "std": 0.1
  }
}
```

Effect: `noisy_image = clamp(original + noise, 0, 1)`

Use for: Testing defenses against gradient-based attacks, evaluating aggregation robustness

---

### Brightness (Images)

```json
{
  "type": "brightness",
  "params": {
    "factor": 0.3
  }
}
```

Effect: `adjusted = clamp(original * factor, 0, 1)`

Use for: Testing robustness to lighting variations, evaluating defense sensitivity to subtle attacks

---

### Token Replacement (NLP)

```json
{
  "type": "token_replacement",
  "params": {
    "replacement_prob": 0.2,
    "vocab_size": 30522
  }
}
```

Use for: Testing NLP model robustness, studying text-based Byzantine attacks

---

## 🧪 Example Configurations

### Delayed Attack

Test if attacks bypass detection when introduced late:

```json
{
  "shared_settings": {
    "dataset_keyword": "femnist_iid",
    "num_of_rounds": 20,
    "aggregation_strategy_keyword": "krum",
    "num_of_clients": 10
  },
  "simulation_strategies": [
    {
      "dynamic_attacks": {
        "enabled": true,
        "schedule": [
          {
            "start_round": 15,
            "end_round": 20,
            "selection_strategy": "specific",
            "client_ids": [0, 1, 2],
            "attack_config": {
              "type": "label_flipping",
              "params": { "flip_fraction": 0.8, "num_classes": 62 }
            }
          }
        ]
      }
    }
  ]
}
```

---

### Escalating Intensity

Find defense failure threshold:

```json
{
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 1,
        "end_round": 5,
        "selection_strategy": "specific",
        "client_ids": [0, 1],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.2, "num_classes": 62 }
        }
      },
      {
        "start_round": 6,
        "end_round": 10,
        "selection_strategy": "specific",
        "client_ids": [0, 1],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.5, "num_classes": 62 }
        }
      },
      {
        "start_round": 11,
        "end_round": 15,
        "selection_strategy": "specific",
        "client_ids": [0, 1],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.8, "num_classes": 62 }
        }
      }
    ]
  }
}
```

---

## 📊 Viewing Results

After running a simulation, results are saved in `out/`:

```bash
out/
└── MM-DD-YYYY_HH-MM-SS/
    ├── csv/
    │   ├── per_client_metrics_0.csv
    │   └── round_metrics_0.csv
    ├── accuracy_history_0.pdf
    ├── loss_history_0.pdf
    └── strategy_config_0.json
```

**Key metrics to analyze:**

- **Per-client accuracy** - Look for drops during attack rounds
- **Krum scores** - Higher scores indicate outlier detection
- **Removal history** - Track which clients were excluded

---

## 🔧 Configuration Schema

**Complete schema:**

```json
{
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 1,
        "end_round": 10,
        "selection_strategy": "specific",
        "client_ids": [0, 1],
        "attack_config": {
          "type": "label_flipping",
          "params": {
            "flip_fraction": 0.5,
            "num_classes": 62,
            "target_class": null
          }
        }
      }
    ]
  }
}
```

**Constraints:**

- `start_round`: 1 ≤ start_round ≤ num_of_rounds
- `end_round`: start_round ≤ end_round ≤ num_of_rounds
- `flip_fraction`: 0.0 ≤ flip_fraction ≤ 1.0
- `percentage`: 0.0 < percentage ≤ 1.0
- `std`: std ≥ 0.0
- `factor`: factor ≥ 0.0

---

## 💡 Troubleshooting

**Attack not triggering:**

1. Check `enabled: true`
2. Verify `start_round` ≤ current ≤ `end_round`
3. Verify client ID in `client_ids` (for "specific")
4. Check logs for "should_poison_this_round" calls

**Simulation crashes:**

1. `num_classes` must match dataset (62 for FEMNIST, 10 for MNIST)
2. No negative parameter values
3. Compatible attack type (don't use `token_replacement` on images)

**Import errors:**

```bash
# Ensure attack_utils is importable
cd src
python -c "from attack_utils.poisoning import should_poison_this_round"
```

---

## 📖 Implementation Details

### Module Structure

```text
src/attack_utils/
├── __init__.py
└── poisoning.py
```

### Key Functions

**`should_poison_this_round(current_round, client_id, attack_schedule)`**

Returns `(bool, dict)` indicating if poisoning should occur and the attack config.

**`apply_poisoning_attack(data, labels, attack_config)`**

Applies the specified attack to the batch data.

**Attack-specific functions:**

- `apply_label_flipping(labels, flip_fraction, num_classes, target_class)`
- `apply_gaussian_noise(images, mean, std)`
- `apply_brightness_attack(images, factor)`
- `apply_token_replacement(tokens, replacement_prob, vocab_size)`

---

## 🚀 Next Steps

1. **Experiment** - Try different attack schedules and parameters
2. **Analyze** - Review metrics in `out/*/csv/` files
3. **Extend** - Add custom attack types in `src/attack_utils/poisoning.py`
4. **Compare** - Test different aggregation strategies against attacks

---

## 📖 References

- **Flower Framework:** [https://flower.ai/](https://flower.ai/)
- **Byzantine Attacks:** Research on adversarial federated learning
- **Test Config:** `config/simulation_strategies/test_dynamic_poisoning.json` - Validated working example with label flipping (rounds 3-8, 70% flip fraction, clients 0-1-2)
