# 🧪 HuggingFace Dataset Loading

**Technical guide for using HuggingFace datasets in federated learning simulations.**

> 📚 **Related Docs:** [README](./README.md) | [attacks.md](./attacks.md)

---

## ⚡ Quick Start

### **Web UI (Recommended)**

1. Navigate to **New Simulation** (`http://localhost:5173/simulations/new`)
2. Select Dataset Source: **HuggingFace Hub**
3. Enter dataset name (e.g., `ylecun/mnist`)
4. Choose partitioning strategy (IID, Dirichlet, or Pathological)
5. Get **real-time validation** with dataset metadata! ✅

### **CLI**

```bash
# Use config files in config/simulation_strategies/examples/
python -m src.simulation_runner  # Will prompt for config selection
```

### **Features** ✨

- ✅ **Real-time validation** - Instant feedback on dataset compatibility
- ✅ **Metadata preview** - View splits, example counts, and key features
- ✅ **Autocomplete** - Popular datasets suggested automatically
- ✅ **Error handling** - Clear error messages for 404, network, and auth issues

---

## 🎯 Configuration Options

### 1. Select Dataset Source

**Web UI:**

- Dataset Source: `HuggingFace Hub`

**Config file:**

```json
{
  "dataset_source": "huggingface",
  "hf_dataset_name": "ylecun/mnist",
  "partitioning_strategy": "iid"
}
```

### 2. Common Datasets

```text
ylecun/mnist          - 28x28 grayscale digits (60k train, 10k test)
uoft-cs/cifar10       - 32x32 RGB images, 10 classes
uoft-cs/cifar100      - 32x32 RGB images, 100 classes
flwrlabs/femnist      - Federated handwritten characters (62 classes)
medmnist/bloodmnist   - Blood cell microscopy images
medmnist/pneumoniamnist - Chest X-ray pneumonia detection
```

Browse: [huggingface.co/datasets](https://huggingface.co/datasets)

### 3. Partitioning Strategies

**IID:** Balanced distribution (unrealistic baseline)
**Dirichlet:** Heterogeneous distribution (realistic, tune α)
**Pathological:** Extreme non-IID (stress testing)

---

## 🔧 Partitioning Details

### IID (Independent and Identically Distributed)

Balanced data across all clients.

**Example:** 10 clients, MNIST

- Each client: ~6,000 images
- All 10 digit classes equally represented
- Unrealistic but useful baseline

**Config:**

```json
{
  "dataset_source": "huggingface",
  "hf_dataset_name": "ylecun/mnist",
  "partitioning_strategy": "iid"
}
```

---

### Dirichlet (Heterogeneous Distribution)

Realistic non-uniform data distribution.

**Example:** 10 clients, CIFAR-10, α=0.5

- Some clients have more "cat" images
- Others have more "dog" images
- Simulates real-world scenarios (hospitals, mobile devices)

**Alpha (α) Parameter:**

- α = 0.1: Very heterogeneous (medical data)
- α = 0.5: Moderate (default)
- α = 1.0: Mild heterogeneity
- α = 10.0: Nearly IID

**Typical research values:** 0.1 - 1.0

**Config:**

```json
{
  "dataset_source": "huggingface",
  "hf_dataset_name": "uoft-cs/cifar10",
  "partitioning_strategy": "dirichlet",
  "partitioning_params": { "alpha": 0.5 }
}
```

---

### Pathological (Extreme Non-IID)

Extreme heterogeneity for stress testing.

**Example:** 10 clients, MNIST

- Client 1: Only digits 0, 1
- Client 2: Only digits 2, 3
- Client 3: Only digits 4, 5

**Use cases:**

- Testing robustness of aggregation strategies
- Extreme data silos
- Byzantine attack research

**Config:**

```json
{
  "dataset_source": "huggingface",
  "hf_dataset_name": "ylecun/mnist",
  "partitioning_strategy": "pathological"
}
```

---

## 🎯 Example Simulations

### Example 1: MNIST with IID

**Goal:** Baseline federated learning

```json
{
  "shared_settings": {
    "dataset_source": "huggingface",
    "hf_dataset_name": "ylecun/mnist",
    "partitioning_strategy": "iid",
    "num_of_clients": 10,
    "num_of_rounds": 5,
    "model_type": "cnn",
    "batch_size": 32
  }
}
```

**Expected:** ~95% accuracy in 5 rounds

---

### Example 2: CIFAR-10 with Dirichlet

**Goal:** Study non-IID impact on convergence

```json
{
  "shared_settings": {
    "dataset_source": "huggingface",
    "hf_dataset_name": "uoft-cs/cifar10",
    "partitioning_strategy": "dirichlet",
    "partitioning_params": { "alpha": 0.5 },
    "num_of_clients": 20,
    "num_of_rounds": 10,
    "model_type": "cnn",
    "batch_size": 64
  }
}
```

**Expected:** ~70-80% accuracy (slower than IID)

**Research Question:** How does α affect convergence speed?

---

### Example 3: FEMNIST with Pathological + Attack

**Goal:** Test Byzantine-robust aggregation under extreme heterogeneity

```json
{
  "shared_settings": {
    "dataset_source": "huggingface",
    "hf_dataset_name": "flwrlabs/femnist",
    "partitioning_strategy": "pathological",
    "num_of_clients": 100,
    "num_of_rounds": 20,
    "num_of_malicious_clients": 20,
    "aggregation_strategy_keyword": "krum"
  },
  "simulation_strategies": [{ "attack_type": "label_flipping" }]
}
```

**Expected:** Study defense effectiveness

---

## 🧪 Heterogeneity Comparison

Run same simulation with different α values:

| α Value | Description        | Expected Convergence |
| ------- | ------------------ | -------------------- |
| 0.1     | Very heterogeneous | Slowest              |
| 0.5     | Moderate           | Medium               |
| 1.0     | Mild               | Faster               |
| 10.0    | Nearly IID         | Fastest              |

**Research Question:** At what α threshold does FedAvg performance degrade?

---

## 🔧 Troubleshooting

### Slow dataset download

**Cause:** First-time download from HuggingFace Hub

**Solution:** Wait for initial download (dataset cached for future use)

---

### "Dataset not found" error

**Cause:** Incorrect dataset name

**Solution:**

1. Verify dataset exists: [huggingface.co/datasets](https://huggingface.co/datasets)
2. Check exact name (case-sensitive): `ylecun/mnist` not `MNIST`
3. Ensure Flower Datasets compatibility

---

### Memory error during partitioning

**Cause:** Too many clients or large dataset

**Solution:**

1. Reduce `num_of_clients` (10-20 instead of 100)
2. Use smaller dataset (MNIST instead of CIFAR-100)
3. Increase available RAM

---

## 🚀 Advanced Usage

### Combining with Dynamic Poisoning Attacks

HuggingFace datasets work seamlessly with dynamic attack scheduling for round-based poisoning:

```json
{
  "dataset_source": "huggingface",
  "hf_dataset_name": "uoft-cs/cifar10",
  "partitioning_strategy": "dirichlet",
  "partitioning_params": { "alpha": 0.3 },
  "aggregation_strategy_keyword": "trimmed_mean",
  "dynamic_attacks": {
    "enabled": true,
    "schedule": [
      {
        "start_round": 5,
        "end_round": 12,
        "selection_strategy": "specific",
        "client_ids": [0, 1, 2],
        "attack_config": {
          "type": "label_flipping",
          "params": { "flip_fraction": 0.6, "num_classes": 10 }
        }
      }
    ]
  }
}
```

**Research Question:** Does heterogeneity make attacks more/less effective?

**See:** [attacks.md](./attacks.md) for complete attack configuration guide

---

### Comparing Aggregation Strategies

Run simulations with different strategies on same dataset:

1. FedAvg + IID (baseline)
2. FedAvg + Dirichlet α=0.5 (realistic)
3. Krum + Dirichlet α=0.5 (robust)

**Compare:** Accuracy, convergence speed, attack resilience

---

## 📝 Research Workflow Example

### Research Question

"How does data heterogeneity affect Byzantine attack success rate?"

### Experimental Setup

1. **Dataset:** MNIST (ylecun/mnist)
2. **Partitioning:** Dirichlet with α ∈ {0.1, 0.5, 1.0, 10.0}
3. **Attack:** Label flipping (20% malicious clients)
4. **Aggregation:** FedAvg vs Krum vs Trimmed Mean

### Steps

1. Run 4 α values × 3 aggregation strategies = 12 simulations
2. Compare final accuracy across α values
3. Plot: α vs accuracy for each strategy
4. Analyze: Does heterogeneity help or hurt attack detection?

---

## 🎓 Key Concepts

### Non-IID Data

**Why it matters:** Real-world FL always has non-IID data (hospitals, user behavior, device types)

**Paper:** "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification" (2019)

### Dirichlet Distribution

Controls label distribution heterogeneity:

- Lower α: More skewed (realistic)
- Higher α: More uniform (closer to IID)

### Pathological vs Dirichlet

- **Pathological:** Deterministic, extreme separation
- **Dirichlet:** Probabilistic, tunable heterogeneity
- **Research tip:** Use Dirichlet (realistic), Pathological (stress testing)

---

## 🚀 Running Simulations

### **Quick Start**

```bash
# From project root - starts both API and frontend
./run_frontend.sh
```

This automatically:

- ✅ Starts API server (port 8000)
- ✅ Starts frontend dev server (port 5173)
- ✅ Opens browser to `http://localhost:5173`

Navigate to **New Simulation** to configure and launch your HuggingFace dataset experiment!

### **Manual Setup**

See [README.md](./README.md#-running-the-application) for detailed instructions.

---

## 📖 References

- **Frontend:** [README.md](./README.md) - Tech stack and features
- **Attacks:** [attacks.md](./attacks.md) - Dynamic poisoning attacks
- **Flower Datasets:** [https://flower.ai/docs/datasets/](https://flower.ai/docs/datasets/)
- **HuggingFace Hub:** [https://huggingface.co/datasets](https://huggingface.co/datasets)
- **Research:** [Understanding Non-IID Data in Federated Learning](https://arxiv.org/abs/1909.06335)
