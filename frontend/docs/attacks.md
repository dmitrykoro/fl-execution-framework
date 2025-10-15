# ⚔️ Dynamic Poisoning Attacks

**Round-based attack scheduling for federated learning simulations.**

> 📚 **Related Docs:** [README](./README.md) | [datasets.md](./datasets.md)

---

## ⚡ Quick Start

### **Web UI (Recommended)**

1. Navigate to **New Simulation** (`http://localhost:5173/simulations/new`)
2. Scroll to **Attack Configuration** section
3. Enable **Dynamic Attacks** toggle
4. Configure attack schedule with:
   - Start/End rounds
   - Client selection strategy
   - Attack type and parameters
5. Get **real-time validation** ensuring attack config is valid! ✅

### **CLI**

**Config:**

```json
{
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
          "params": { "flip_fraction": 0.5, "num_classes": 10 }
        }
      }
    ]
  }
}
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

### What is a Batch?

A **batch** is a small subset of training data processed together. Instead of training on one example at a time (slow) or all examples at once (memory-intensive), data is split into chunks.

**Example:** 1,000 images with `batch_size = 32`:

- **Batch 1**: Images 1-32
- **Batch 2**: Images 33-64
- **Batch 3**: Images 65-96
- ... and so on

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

See implementation in:

- `src/attack_utils/poisoning.py:103-149` - Schedule checking logic
- `src/client_models/flower_client.py:97-104` - Batch-level poisoning for CNNs
- `src/client_models/flower_client.py:140-148` - Batch-level poisoning for transformers

---

## 🕒 Schedule Configuration

**Basic:**

```json
{
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 1,        // Inclusive
        "end_round": 10,         // Inclusive
        "selection_strategy": "specific",
        "client_ids": [0, 1, 2],
        "attack_config": {
          "type": "label_flipping",
          "params": { ... }
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
        "client_ids": [0],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.8, "num_classes": 10 }
        }
      },
      {
        "start_round": 10,
        "end_round": 15,
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

**Specific clients:**

```json
{
  "selection_strategy": "specific",
  "client_ids": [0, 1, 3, 7]
}
```

Use for: Studying specific client behavior in heterogeneous settings

**Random selection:**

```json
{
  "selection_strategy": "random",
  "num_clients": 3,
  "_selected_clients": [2, 5, 7] // Framework sets at runtime
}
```

Use for: Simulating unpredictable adversaries

**Percentage-based:**

```json
{
  "selection_strategy": "percentage",
  "percentage": 0.2, // 20% of clients
  "_selected_clients": [1, 4, 6, 9] // Framework sets
}
```

Use for: Scaling attacks with total client count

---

## 💣 Attack Types

### Label Flipping (Classification)

**Random:**

```json
{
  "type": "label_flipping",
  "params": {
    "flip_fraction": 0.5, // Flip 50% of labels
    "num_classes": 10
  }
}
```

**Targeted:**

```json
{
  "type": "label_flipping",
  "params": {
    "flip_fraction": 1.0,
    "num_classes": 10,
    "target_class": 7 // All → class 7
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
    "std": 0.1 // Higher = more noise
  }
}
```

**Effect:** `noisy_image = clamp(original + noise, 0, 1)`

Use for: Testing defenses against gradient-based attacks, evaluating aggregation robustness

---

### Brightness (Images)

```json
{
  "type": "brightness",
  "params": {
    "factor": 0.3 // 0.0 = black, 1.0 = unchanged, >1.0 = brighter
  }
}
```

**Effect:** `adjusted = clamp(original * factor, 0, 1)`

Use for: Testing robustness to lighting variations, evaluating defense sensitivity to subtle attacks

---

### Token Replacement (NLP)

```json
{
  "type": "token_replacement",
  "params": {
    "replacement_prob": 0.2, // Replace 20% of tokens
    "vocab_size": 30522 // BERT default
  }
}
```

Use for: Testing NLP model robustness, studying text-based Byzantine attacks

---

## 🧪 Examples

### Delayed Attack

Test if attacks bypass detection when introduced late:

```json
{
  "hf_dataset_name": "ylecun/mnist",
  "num_of_rounds": 20,
  "aggregation_strategy_keyword": "krum",
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 15, // Late attack
        "end_round": 20,
        "client_ids": [0, 1, 2],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.8, "num_classes": 10 }
        }
      }
    ]
  }
}
```

---

### Multi-Phase Attack

Test defense adaptation to changing attack patterns:

```json
{
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 1,
        "end_round": 5,
        "client_ids": [0],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.5, "num_classes": 10 }
        }
      },
      {
        "start_round": 10,
        "end_round": 15,
        "client_ids": [1, 2],
        "attack_config": {
          "type": "brightness",
          "params": { "factor": 0.3 }
        }
      }
    ]
  }
}
```

---

### Heterogeneity Impact

Test attack effectiveness under data heterogeneity:

```json
{
  "hf_dataset_name": "uoft-cs/cifar10",
  "partitioning_strategy": "dirichlet",
  "partitioning_params": { "alpha": 0.3 },
  "aggregation_strategy_keyword": "trimmed_mean",
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 3,
        "end_round": 12,
        "selection_strategy": "percentage",
        "percentage": 0.3,
        "attack_config": {
          "type": "gaussian_noise",
          "params": { "mean": 0.0, "std": 0.15 }
        }
      }
    ]
  }
}
```

**Sweep:** α ∈ {0.1, 0.5, 1.0, 10.0}

---

### Escalating Intensity

Find defense failure threshold:

```json
{
  "dynamic_attacks": {
    "schedule": [
      {
        "start_round": 1,
        "end_round": 5,
        "client_ids": [0, 1],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.2, "num_classes": 10 }
        }
      },
      {
        "start_round": 6,
        "end_round": 10,
        "client_ids": [0, 1],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.5, "num_classes": 10 }
        }
      },
      {
        "start_round": 11,
        "end_round": 15,
        "client_ids": [0, 1],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.8, "num_classes": 10 }
        }
      }
    ]
  }
}
```

---

## 🔬 HuggingFace Integration

Dynamic attacks work with HuggingFace datasets:

```json
{
  "dataset_source": "huggingface",
  "hf_dataset_name": "ylecun/mnist",
  "partitioning_strategy": "dirichlet",
  "partitioning_params": { "alpha": 0.5 },
  "aggregation_strategy_keyword": "krum",
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 5,
        "end_round": 12,
        "client_ids": [0, 1, 2, 3],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.6, "num_classes": 10 }
        }
      }
    ]
  }
}
```

See [datasets.md](./datasets.md) for details.

---

## 📊 Experimental Design

**Control groups:**

```json
{ "dynamic_attacks": { "enabled": false } } // Baseline
```

**Attack timing:**

- Early: Rounds 1-5 (disrupts initialization)
- Mid: Rounds 5-10 (affects convergence)
- Late: Rounds 15-20 (tests stability)

**Multiple runs:** 3-5 runs per configuration with different seeds

**Parameter sweeps:** Test flip_fraction ∈ {0.2, 0.4, 0.6, 0.8, 1.0}

---

## 🔧 Configuration Reference

**Complete schema:**

```json
{
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 1, // Required (1-indexed)
        "end_round": 10, // Required (inclusive)
        "selection_strategy": "specific", // "specific" | "random" | "percentage"
        "client_ids": [0, 1], // Required for "specific"
        "num_clients": 3, // Required for "random"
        "percentage": 0.2, // Required for "percentage"
        "attack_config": {
          "type": "label_flipping", // Attack type
          "params": {
            "flip_fraction": 0.5,
            "num_classes": 10,
            "target_class": null // Optional
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

**Simulation crashes:**

1. `num_classes` must match dataset (10 for MNIST/CIFAR-10, 100 for CIFAR-100)
2. No negative parameter values
3. Compatible attack type (don't use `token_replacement` on images)

**Inconsistent results:**

- Use fixed random seed
- Run 3-5 trials
- Check CPU/memory constraints

---

## 🚀 Running Simulations

### **Quick Start**

```bash
# From project root - starts both API and frontend
./start-dev.sh
```

This automatically:

- ✅ Starts API server (port 8000)
- ✅ Starts frontend dev server (port 5173)
- ✅ Opens browser to `http://localhost:5173`

Navigate to **New Simulation** → **Attack Configuration** to configure dynamic attacks!

### **Manual Setup**

See [README.md](./README.md#-running-the-application) for detailed instructions.

---

## 📖 References

- **Frontend:** [README.md](./README.md) - Tech stack and features
- **Datasets:** [datasets.md](./datasets.md) - HuggingFace loading and partitioning
- **Flower:** [https://flower.ai/](https://flower.ai/)
- **Research:** Byzantine attacks and defenses in federated learning
