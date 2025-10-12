"""
Dataset Inspector for HuggingFace Datasets.

Automatically extracts metadata from HuggingFace datasets to enable dynamic
model configuration and dataset-agnostic federated learning.
"""

from datasets import load_dataset_builder, load_dataset
from typing import Dict, List, Optional, Tuple
import logging


class DatasetInspector:
    """Automatically inspect HuggingFace datasets to extract metadata."""

    @staticmethod
    def inspect_dataset(dataset_name: str, label_column: str = "label") -> Dict:
        """
        Inspect a HuggingFace dataset and extract comprehensive metadata.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "flwrlabs/femnist")
            label_column: Name of the label/target column (default: "label")

        Returns:
            Dictionary containing:
                - modality: 'image' | 'text' | 'multimodal'
                - num_classes: Number of classification labels
                - class_names: List of class label names
                - image_column: Name of the image column (if exists)
                - image_shape: Tuple (channels, height, width) for images
                - text_columns: List of text column names
                - features: Raw HuggingFace features dict
        """
        try:
            builder = load_dataset_builder(dataset_name)
            features = builder.info.features

            metadata = {
                "features": features,
                "num_classes": None,
                "class_names": [],
                "modality": None,
                "image_column": None,
                "image_shape": None,
                "text_columns": [],
            }

            # Detect label column and extract class info
            if label_column in features:
                label_feature = features[label_column]
                if hasattr(label_feature, "names"):
                    metadata["num_classes"] = len(label_feature.names)
                    metadata["class_names"] = label_feature.names
                    logging.info(
                        f"Detected {metadata['num_classes']} classes in '{label_column}' column"
                    )

            # Detect modality by inspecting feature types
            has_image = DatasetInspector._has_image_column(features)
            has_text = DatasetInspector._has_text_column(features)

            if has_image and has_text:
                metadata["modality"] = "multimodal"
            elif has_image:
                metadata["modality"] = "image"
            elif has_text:
                metadata["modality"] = "text"
            else:
                metadata["modality"] = "unknown"

            # Extract image column and infer shape
            if has_image:
                img_col, img_shape = DatasetInspector._get_image_info(
                    dataset_name, features
                )
                metadata["image_column"] = img_col
                metadata["image_shape"] = img_shape
                logging.info(
                    f"Detected image column '{img_col}' with shape {img_shape}"
                )

            # Extract text columns
            metadata["text_columns"] = DatasetInspector._get_text_columns(features)
            if metadata["text_columns"]:
                logging.info(f"Detected text columns: {metadata['text_columns']}")

            logging.info(
                f"Dataset '{dataset_name}' inspection complete: "
                f"modality={metadata['modality']}, "
                f"classes={metadata['num_classes']}"
            )

            return metadata

        except Exception as e:
            logging.error(f"Failed to inspect dataset '{dataset_name}': {e}")
            raise

    @staticmethod
    def _has_image_column(features: Dict) -> bool:
        """Check if features contain an Image column."""
        for feature in features.values():
            feature_type = str(type(feature))
            if "Image" in feature_type:
                return True
        return False

    @staticmethod
    def _has_text_column(features: Dict) -> bool:
        """Check if features contain common text columns."""
        text_column_names = [
            "text",
            "sentence",
            "sentence1",
            "sentence2",
            "question",
            "answer",
            "premise",
            "hypothesis",
            "document",
            "passage",
        ]
        return any(col in features for col in text_column_names)

    @staticmethod
    def _get_image_info(
        dataset_name: str, features: Dict
    ) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
        """
        Extract image column name and shape.

        Returns:
            Tuple of (column_name, (channels, height, width))
        """
        # Find image column
        image_column = None
        for col_name, feature in features.items():
            if "Image" in str(type(feature)):
                image_column = col_name
                break

        if not image_column:
            return None, None

        # Known dataset configurations (avoids slow streaming)
        known_datasets = {
            "flwrlabs/femnist": (1, 28, 28),
            "mnist": (1, 28, 28),
            "fashion_mnist": (1, 28, 28),
            "cifar10": (3, 32, 32),
            "cifar100": (3, 32, 32),
        }

        # Check if this is a known dataset
        if dataset_name in known_datasets:
            img_shape = known_datasets[dataset_name]
            logging.info(
                f"Using cached dimensions for known dataset '{dataset_name}': {img_shape}"
            )
            return image_column, img_shape

        try:
            # For unknown datasets, load just one sample (non-streaming with limit)
            sample_ds = load_dataset(dataset_name, split="train[:1]")
            first_sample = sample_ds[0]
            img = first_sample[image_column]

            # Convert PIL image to tensor shape: (C, H, W)
            if hasattr(img, "mode") and hasattr(img, "size"):
                # PIL Image
                channels = 3 if img.mode == "RGB" else 1
                width, height = img.size
                return image_column, (channels, height, width)
            else:
                # Already a tensor or array
                logging.warning(
                    "Image format not recognized, using default shape (1, 28, 28)"
                )
                return image_column, (1, 28, 28)

        except Exception as e:
            logging.warning(
                f"Could not infer image shape: {e}. Using default (1, 28, 28)"
            )
            return image_column, (1, 28, 28)

    @staticmethod
    def _get_text_columns(features: Dict) -> List[str]:
        """Extract all text column names."""
        text_columns = []
        common_text_cols = [
            "text",
            "sentence",
            "sentence1",
            "sentence2",
            "question",
            "answer",
            "premise",
            "hypothesis",
            "document",
            "passage",
        ]

        for col_name in common_text_cols:
            if col_name in features:
                text_columns.append(col_name)

        return text_columns
