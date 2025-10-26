# 📊 Attack Snapshot Analysis Guide

Programmatic analysis of attack snapshot data using Python.

---

## 📋 Quick Reference

| Task | Tool | Format |
|------|------|--------|
| Visual inspection | PNG files | No code needed |
| Check metadata | JSON files | Text editor |
| Statistical analysis | Pickle files | Python + this guide |
| Export to Excel/R | Pickle → CSV | pandas |

---

## 🔍 Understanding Pickle Files

### What is Pickle?

Python's built-in serialization format for saving objects to disk.

**Why we use pickle:**

- Preserves exact data types (no conversion loss)
- Fast save/load operations
- Single file for complex data structures
- Includes full numerical arrays
- No extra dependencies

**When to use pickle vs other formats:**

| Format | Use Case |
|--------|----------|
| PNG | Quick visual inspection |
| JSON | Check attack configuration |
| Pickle | Statistical analysis, pixel-level verification |

---

## 📊 Loading and Inspecting Snapshots

### Print Metadata

```python
from src.attack_utils.attack_snapshots import load_attack_snapshot

snapshot = load_attack_snapshot(
    "out/<run_id>/attack_snapshots/client_0/round_3/label_flipping.pickle"
)

print("=== Attack Snapshot Info ===")
print(f"Client ID: {snapshot['metadata']['client_id']}")
print(f"Round: {snapshot['metadata']['round_num']}")
print(f"Attack Type: {snapshot['metadata']['attack_type']}")
print(f"Number of Samples: {snapshot['metadata']['num_samples']}")
print(f"Data Shape: {snapshot['metadata']['data_shape']}")
print(f"\nAttack Config:")
for key, value in snapshot['metadata']['attack_config'].items():
    print(f"  {key}: {value}")
```

### Visualize Images with Matplotlib

```python
import matplotlib.pyplot as plt
from src.attack_utils.attack_snapshots import load_attack_snapshot

snapshot = load_attack_snapshot(
    "out/<run_id>/attack_snapshots/client_0/round_3/label_flipping.pickle"
)

# Extract data
images = snapshot["data"]          # Shape: (N, C, H, W)
labels = snapshot["labels"]        # Shape: (N,)
original_labels = snapshot.get("original_labels", labels)

# Plot grid
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle(f"Poisoned Samples - Client {snapshot['metadata']['client_id']}, Round {snapshot['metadata']['round_num']}")

for i, ax in enumerate(axes):
    # Handle grayscale (C=1) or RGB (C=3)
    if images.shape[1] == 1:
        ax.imshow(images[i, 0], cmap='gray')
    else:
        ax.imshow(images[i].transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)

    # Show labels
    if original_labels is not None:
        ax.set_title(f"Label: {labels[i]}\n(was {original_labels[i]})")
    else:
        ax.set_title(f"Label: {labels[i]}")

    ax.axis('off')

plt.tight_layout()
plt.savefig("poisoned_samples.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Compare Before/After Attack

```python
import matplotlib.pyplot as plt
from src.attack_utils.attack_snapshots import load_attack_snapshot

# Load two snapshots
pre_attack = load_attack_snapshot("out/<run_id>/attack_snapshots/client_0/round_2/label_flipping.pickle")
during_attack = load_attack_snapshot("out/<run_id>/attack_snapshots/client_0/round_3/label_flipping.pickle")

# Compare first image
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(pre_attack["data"][0, 0], cmap='gray')
axes[0].set_title(f"Round 2 (Normal)\nLabel: {pre_attack['labels'][0]}")
axes[0].axis('off')

axes[1].imshow(during_attack["data"][0, 0], cmap='gray')
axes[1].set_title(f"Round 3 (Poisoned)\nLabel: {during_attack['labels'][0]}")
axes[1].axis('off')

plt.suptitle(f"Attack Type: {during_attack['metadata']['attack_type']}")
plt.tight_layout()
plt.show()
```

---

## 📈 Statistical Analysis

### Analyze Label Distribution

```python
import numpy as np
from collections import Counter
from src.attack_utils.attack_snapshots import list_attack_snapshots, load_attack_snapshot

# List all snapshots for a run
snapshots = list_attack_snapshots("out/<run_id>")

# Collect all labels
all_labels = []
all_original_labels = []

for snapshot_path in snapshots:
    snapshot = load_attack_snapshot(str(snapshot_path))
    all_labels.extend(snapshot["labels"].tolist())
    if "original_labels" in snapshot:
        all_original_labels.extend(snapshot["original_labels"].tolist())

# Count distributions
print("=== Label Distribution Analysis ===")
print(f"Total samples: {len(all_labels)}")
print(f"\nPoisoned labels:")
print(Counter(all_labels))

if all_original_labels:
    print(f"\nOriginal labels:")
    print(Counter(all_original_labels))

    # Count flips
    flips = sum(1 for orig, new in zip(all_original_labels, all_labels) if orig != new)
    print(f"\nLabels flipped: {flips}/{len(all_labels)} ({100*flips/len(all_labels):.1f}%)")
```

### Batch Summary

```python
from src.attack_utils.attack_snapshots import get_snapshot_summary

summary = get_snapshot_summary("out/<run_id>")

print("=== Attack Summary ===")
print(f"Total snapshots: {summary['total_snapshots']}")
print(f"Clients attacked: {summary['clients_attacked']}")
print(f"Rounds with attacks: {summary['rounds_with_attacks']}")
print(f"Attack types used: {summary['attack_types']}")
```

---

## 📤 Export to Other Formats

### Export Images to PNG

```python
from PIL import Image
import numpy as np
from src.attack_utils.attack_snapshots import load_attack_snapshot

snapshot = load_attack_snapshot("out/<run_id>/attack_snapshots/client_0/round_3/label_flipping.pickle")

for i, img_array in enumerate(snapshot["data"]):
    # Convert to PIL Image
    if img_array.shape[0] == 1:  # Grayscale
        img = Image.fromarray((img_array[0] * 255).astype(np.uint8), mode='L')
    else:  # RGB
        img = Image.fromarray((img_array.transpose(1, 2, 0) * 255).astype(np.uint8))

    filename = f"sample_{i}_label_{snapshot['labels'][i]}.png"
    img.save(filename)
    print(f"Saved {filename}")
```

### Export Metadata to JSON

```python
import json
from src.attack_utils.attack_snapshots import load_attack_snapshot

snapshot = load_attack_snapshot("out/<run_id>/attack_snapshots/client_0/round_3/label_flipping.pickle")

with open("snapshot_metadata.json", "w") as f:
    json.dump(snapshot["metadata"], f, indent=2)

print("Saved metadata to snapshot_metadata.json")
```

### Export to CSV (Excel/R)

```python
import pandas as pd
from src.attack_utils.attack_snapshots import load_attack_snapshot, list_attack_snapshots

# Collect data from all snapshots
records = []

for snapshot_path in list_attack_snapshots("out/<run_id>"):
    snapshot = load_attack_snapshot(str(snapshot_path))

    for i in range(len(snapshot["labels"])):
        records.append({
            "client_id": snapshot["metadata"]["client_id"],
            "round": snapshot["metadata"]["round_num"],
            "attack_type": snapshot["metadata"]["attack_type"],
            "sample_index": i,
            "poisoned_label": snapshot["labels"][i],
            "original_label": snapshot.get("original_labels", [None]*len(snapshot["labels"]))[i],
            "data_shape": str(snapshot["metadata"]["data_shape"])
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(records)
df.to_csv("attack_snapshots_summary.csv", index=False)
print(f"Exported {len(records)} samples to attack_snapshots_summary.csv")
```

---

## 🔧 Common Workflow

```python
from src.attack_utils.attack_snapshots import load_attack_snapshot

# 1. Load snapshot
snapshot = load_attack_snapshot("out/<run_id>/attack_snapshots/<path>")

# 2. Verify attack applied correctly
print(snapshot["metadata"]["attack_config"])

# 3. Visualize or analyze
# ... matplotlib/pandas code ...

# 4. Export for external tools
# ... CSV/PNG export code ...
```

---

## 📚 API Reference

### `load_attack_snapshot(filepath: str) -> dict`

Load pickle snapshot from disk.

**Returns:**

```python
{
    "metadata": {
        "client_id": int,
        "round_num": int,
        "attack_type": str,
        "num_samples": int,
        "attack_config": dict,
        "timestamp": str,
        "experiment_info": dict
    },
    "data": np.ndarray,           # Shape: (N, C, H, W)
    "labels": np.ndarray,         # Shape: (N,)
    "original_labels": np.ndarray # Shape: (N,)
}
```

### `list_attack_snapshots(output_dir: str) -> list[Path]`

List all pickle files in snapshot directory (hierarchical structure).

### `get_snapshot_summary(output_dir: str) -> dict`

Get summary statistics across all snapshots.

**Returns:**

```python
{
    "total_snapshots": int,
    "clients_attacked": list[int],
    "rounds_with_attacks": list[int],
    "attack_types": list[str]
}
```
