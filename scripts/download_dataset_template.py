#!/usr/bin/env python3
"""
Template for creating dataset download scripts.

Copy this file and modify it to download and prepare your own datasets.
The framework expects datasets in this structure:
    datasets/your_dataset_name/client_X/label_Y/sample_Z.png

This template shows the common pattern used in the FL execution framework.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def download_and_prepare_dataset(
    output_dir: str = "datasets/my_dataset",
    num_clients: int = 10,
    min_samples_per_client: int = 50,
):
    """
    Download and organize your dataset.

    Args:
        output_dir: Target directory for the dataset
        num_clients: Number of clients to create
        min_samples_per_client: Minimum samples required per client
    """
    try:
        from flwr_datasets import FederatedDataset

        # Import the partitioner you need:
        # - NaturalIdPartitioner: for naturally partitioned data (non-IID)
        # - IidPartitioner: for IID partitioning
        # - DirichletPartitioner: for Dirichlet distribution (non-IID)
        from flwr_datasets.partitioner import IidPartitioner
    except ImportError:
        print("Error: flwr-datasets not installed.")
        print("Install with: pip install flwr-datasets[vision]")
        sys.exit(1)

    print("Downloading dataset...")
    print(f"Target directory: {output_dir}")
    print(f"Number of clients: {num_clients}")

    # TODO: Replace with your dataset from Flower Datasets
    # Browse available datasets at: https://flower.ai/docs/datasets/
    fds = FederatedDataset(
        dataset="your/dataset-name",  # e.g., "flwrlabs/mnist"
        partitioners={"train": IidPartitioner(num_partitions=num_clients)},
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {num_clients} client partitions...")

    # Track statistics
    client_stats = []

    for client_id in range(num_clients):
        try:
            # Load partition for this client
            partition = fds.load_partition(client_id, split="train")

            # Filter clients with too few samples
            if len(partition) < min_samples_per_client:
                print(
                    f"  Skipping client {client_id}: only {len(partition)} samples (min: {min_samples_per_client})"
                )
                continue

            client_dir = output_path / f"client_{client_id}"
            client_dir.mkdir(exist_ok=True)

            # Group images by label
            label_counts = {}

            for idx, sample in enumerate(partition):
                # Debug: print sample keys for first iteration
                if idx == 0:
                    print(f"    Sample keys: {list(sample.keys())}")

                # TODO: Adjust these field names based on your dataset
                image = sample["image"]  # or "img", "pixel_values", etc.
                label = sample["label"]  # or "target", "class", etc.

                # Create label directory if it doesn't exist
                label_dir = client_dir / str(label)
                label_dir.mkdir(exist_ok=True)

                # Save image
                img_filename = label_dir / f"img_{idx}.png"

                # Convert to PIL Image if needed
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                elif not isinstance(image, Image.Image):
                    # Handle other formats (e.g., torch tensor)
                    image = Image.fromarray(np.array(image))

                # TODO: Adjust mode conversion based on your dataset
                # Use "L" for grayscale, "RGB" for color
                if image.mode != "L":
                    image = image.convert("L")

                image.save(img_filename)

                # Track label distribution
                label_counts[label] = label_counts.get(label, 0) + 1

            client_stats.append(
                {
                    "client_id": client_id,
                    "total_samples": len(partition),
                    "num_labels": len(label_counts),
                    "label_dist": label_counts,
                }
            )

            print(
                f"  Client {client_id}: {len(partition)} samples, {len(label_counts)} unique labels"
            )

        except Exception as e:
            print(f"  Error processing client {client_id}: {e}")
            continue

    # Print summary
    print(f"\n{'=' * 60}")
    print("Dataset preparation complete!")
    print(f"{'=' * 60}")
    print(f"Location: {output_dir}")
    print(f"Clients created: {len(client_stats)}")

    if client_stats:
        total_samples = sum(s["total_samples"] for s in client_stats)
        avg_samples = total_samples / len(client_stats)
        print(f"Total samples: {total_samples}")
        print(f"Average samples per client: {avg_samples:.1f}")

        # Show label distribution diversity
        label_diversity = [s["num_labels"] for s in client_stats]
        print("Label diversity (unique labels per client):")
        print(f"  Min: {min(label_diversity)}")
        print(f"  Max: {max(label_diversity)}")
        print(f"  Average: {np.mean(label_diversity):.1f}")

    print(
        f"\nYou can now use this dataset with dataset_keyword='{Path(output_dir).name}'"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/my_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of clients to create",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples per client",
    )

    args = parser.parse_args()

    download_and_prepare_dataset(
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        min_samples_per_client=args.min_samples,
    )
