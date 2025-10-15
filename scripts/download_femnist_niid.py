#!/usr/bin/env python3
"""
Download and prepare FEMNIST Non-IID dataset from Flower Datasets.

This script downloads the FEMNIST dataset using flwr-datasets and organizes it
into the directory structure expected by the FL execution framework:
    datasets/femnist_niid/client_X/label_Y/img_Z.png
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def download_and_prepare_femnist(
    output_dir: str = "datasets/femnist_niid",
    num_clients: int = 100,
    min_samples_per_client: int = 50,
):
    """
    Download FEMNIST dataset and organize by writer (naturally non-IID).

    Args:
        output_dir: Target directory for the dataset
        num_clients: Number of clients to create
        min_samples_per_client: Minimum samples required per client
    """
    try:
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import NaturalIdPartitioner
    except ImportError:
        print("Error: flwr-datasets not installed.")
        print("Install with: pip install flwr-datasets[vision]")
        sys.exit(1)

    print("Downloading FEMNIST dataset...")
    print(f"Target directory: {output_dir}")
    print(f"Number of clients: {num_clients}")

    # Load FEMNIST with natural partitioning by writer_id (non-IID)
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
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

                image = sample["image"]
                # FEMNIST uses "character" field for labels
                label = sample["character"]

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

                # Ensure grayscale
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

    print("\nYou can now use this dataset with dataset_keyword='femnist_niid'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FEMNIST Non-IID dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/femnist_niid",
        help="Output directory for dataset (default: datasets/femnist_niid)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
        help="Number of clients to create (default: 100)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples per client (default: 50)",
    )

    args = parser.parse_args()

    download_and_prepare_femnist(
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        min_samples_per_client=args.min_samples,
    )
