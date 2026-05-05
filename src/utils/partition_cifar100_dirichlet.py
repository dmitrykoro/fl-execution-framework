"""
Partition CIFAR-100 into per-client ImageFolder directories using a Dirichlet distribution.

Usage:
    python src/utils/partition_cifar100_dirichlet.py --alpha 0.1 --num_clients 100 --seed 42
    python src/utils/partition_cifar100_dirichlet.py --alpha 0.5 --num_clients 100 --seed 42
    python src/utils/partition_cifar100_dirichlet.py --alpha 2.0 --num_clients 100 --seed 42

Output structure (consumed by ImageDatasetLoader via torchvision.datasets.ImageFolder):
    datasets/cifar100_a{alpha}/
        client_0/
            apple/  mushroom/  baby/ ... (all 100 class dirs, some may be empty)
        client_1/
            ...
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.datasets import CIFAR100


def alpha_to_tag(alpha: float) -> str:
    """Convert alpha float to directory tag: 0.1 -> 'a01', 0.5 -> 'a05', 2.0 -> 'a20'"""
    return "a" + str(alpha).replace(".", "")


def partition(alpha: float, num_clients: int, out_base: str, seed: int):
    rng = np.random.default_rng(seed)
    tag = alpha_to_tag(alpha)
    out_dir = Path(out_base) / f"cifar100_{tag}"

    print(f"Downloading CIFAR-100...")
    ds = CIFAR100(root="/tmp/cifar100_download", train=True, download=True)

    class_names = ds.classes  # list of 100 class names
    num_classes = len(class_names)

    # Group sample indices by class
    class_indices = [[] for _ in range(num_classes)]
    for idx in range(len(ds)):
        _, label = ds[idx]
        class_indices[label].append(idx)

    # Dirichlet partition: for each class, draw proportions and distribute samples
    # Only create directories for clients that actually receive images for a class.
    # Consistent global label mapping is handled by Cifar100DatasetLoader.
    client_sample_counts = [0] * num_clients
    for cls_idx, indices in enumerate(class_indices):
        proportions = rng.dirichlet([alpha] * num_clients)
        # Convert to integer counts, fixing rounding to avoid losing samples
        counts = (proportions * len(indices)).astype(int)
        counts[-1] = len(indices) - counts[:-1].sum()

        rng.shuffle(indices)  # shuffle within class before assigning
        start = 0
        for c_id, n in enumerate(counts):
            chunk = indices[start: start + n]
            start += n
            cls_name = class_names[cls_idx]
            if len(chunk) == 0:
                continue
            client_dir = out_dir / f"client_{c_id}" / cls_name
            client_dir.mkdir(parents=True, exist_ok=True)
            for img_idx in chunk:
                img, _ = ds[img_idx]
                img_path = client_dir / f"{img_idx}.png"
                if isinstance(img, np.ndarray):
                    Image.fromarray(img).save(img_path)
                else:
                    img.save(img_path)
            client_sample_counts[c_id] += n

    total = sum(client_sample_counts)
    print(f"alpha={alpha} | {num_clients} clients | {total} total samples saved to {out_dir}")
    print(f"  min samples/client: {min(client_sample_counts)}")
    print(f"  max samples/client: {max(client_sample_counts)}")
    print(f"  mean samples/client: {total // num_clients}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True, help="Dirichlet concentration parameter")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--out_base", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    partition(args.alpha, args.num_clients, args.out_base, args.seed)
