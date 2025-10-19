from unittest.mock import Mock, patch

from src.dataset_loaders.dataset_inspector import DatasetInspector
from tests.common import pytest


class TestDatasetInspector:
    """Tests for DatasetInspector."""

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_extracts_basic_metadata(self, mock_builder_fn):
        """Verify inspect_dataset extracts features and initializes metadata."""
        mock_builder = Mock()
        mock_features = {}
        mock_builder.info.features = mock_features
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=False):
            with patch.object(DatasetInspector, "_has_text_column", return_value=False):
                metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["features"] == mock_features
        assert metadata["num_classes"] is None
        assert metadata["class_names"] == []
        assert metadata["modality"] == "unknown"
        assert metadata["image_column"] is None
        assert metadata["image_shape"] is None
        assert metadata["text_columns"] == []

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_detects_num_classes_from_label_names(
        self, mock_builder_fn
    ):
        """Verify num_classes is detected from label feature names."""
        mock_builder = Mock()
        mock_label_feature = Mock()
        mock_label_feature.names = ["cat", "dog", "bird"]
        mock_features = {"label": mock_label_feature}
        mock_builder.info.features = mock_features
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=False):
            with patch.object(DatasetInspector, "_has_text_column", return_value=False):
                metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["num_classes"] == 3
        assert metadata["class_names"] == ["cat", "dog", "bird"]

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_uses_custom_label_column(self, mock_builder_fn):
        """Verify custom label_column parameter is used."""
        mock_builder = Mock()
        mock_label_feature = Mock()
        mock_label_feature.names = ["yes", "no"]
        mock_features = {"sentiment": mock_label_feature, "label": Mock()}
        mock_builder.info.features = mock_features
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=False):
            with patch.object(DatasetInspector, "_has_text_column", return_value=False):
                metadata = DatasetInspector.inspect_dataset(
                    "test/dataset", label_column="sentiment"
                )

        assert metadata["num_classes"] == 2
        assert metadata["class_names"] == ["yes", "no"]

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_handles_missing_label_column(self, mock_builder_fn):
        """Verify graceful handling when label column is missing."""
        mock_builder = Mock()
        mock_features = {"other_column": Mock()}
        mock_builder.info.features = mock_features
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=False):
            with patch.object(DatasetInspector, "_has_text_column", return_value=False):
                metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["num_classes"] is None
        assert metadata["class_names"] == []

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_handles_label_without_names(self, mock_builder_fn):
        """Verify handling when label feature has no names attribute."""
        mock_builder = Mock()
        mock_label_feature = Mock(spec=[])
        mock_features = {"label": mock_label_feature}
        mock_builder.info.features = mock_features
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=False):
            with patch.object(DatasetInspector, "_has_text_column", return_value=False):
                metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["num_classes"] is None
        assert metadata["class_names"] == []

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_detects_image_modality(self, mock_builder_fn):
        """Verify modality is set to 'image' when only images detected."""
        mock_builder = Mock()
        mock_builder.info.features = {}
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=True):
            with patch.object(DatasetInspector, "_has_text_column", return_value=False):
                with patch.object(
                    DatasetInspector,
                    "_get_image_info",
                    return_value=("img", (3, 32, 32)),
                ):
                    with patch.object(
                        DatasetInspector, "_get_text_columns", return_value=[]
                    ):
                        metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["modality"] == "image"

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_detects_text_modality(self, mock_builder_fn):
        """Verify modality is set to 'text' when only text detected."""
        mock_builder = Mock()
        mock_builder.info.features = {}
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=False):
            with patch.object(DatasetInspector, "_has_text_column", return_value=True):
                with patch.object(
                    DatasetInspector, "_get_text_columns", return_value=["text"]
                ):
                    metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["modality"] == "text"

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_detects_multimodal(self, mock_builder_fn):
        """Verify modality is set to 'multimodal' when both image and text detected."""
        mock_builder = Mock()
        mock_builder.info.features = {}
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=True):
            with patch.object(DatasetInspector, "_has_text_column", return_value=True):
                with patch.object(
                    DatasetInspector,
                    "_get_image_info",
                    return_value=("img", (3, 32, 32)),
                ):
                    with patch.object(
                        DatasetInspector, "_get_text_columns", return_value=["text"]
                    ):
                        metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["modality"] == "multimodal"

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_extracts_image_info(self, mock_builder_fn):
        """Verify image column and shape are extracted."""
        mock_builder = Mock()
        mock_builder.info.features = {}
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=True):
            with patch.object(DatasetInspector, "_has_text_column", return_value=False):
                with patch.object(
                    DatasetInspector,
                    "_get_image_info",
                    return_value=("image", (1, 28, 28)),
                ):
                    with patch.object(
                        DatasetInspector, "_get_text_columns", return_value=[]
                    ):
                        metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["image_column"] == "image"
        assert metadata["image_shape"] == (1, 28, 28)

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_extracts_text_columns(self, mock_builder_fn):
        """Verify text columns are extracted."""
        mock_builder = Mock()
        mock_builder.info.features = {}
        mock_builder_fn.return_value = mock_builder

        with patch.object(DatasetInspector, "_has_image_column", return_value=False):
            with patch.object(DatasetInspector, "_has_text_column", return_value=True):
                with patch.object(
                    DatasetInspector,
                    "_get_text_columns",
                    return_value=["sentence1", "sentence2"],
                ):
                    metadata = DatasetInspector.inspect_dataset("test/dataset")

        assert metadata["text_columns"] == ["sentence1", "sentence2"]

    @patch("src.dataset_loaders.dataset_inspector.load_dataset_builder")
    def test_inspect_dataset_raises_on_load_error(self, mock_builder_fn):
        """Verify exception is raised when dataset loading fails."""
        mock_builder_fn.side_effect = Exception("Dataset not found")

        with pytest.raises(Exception, match="Dataset not found"):
            DatasetInspector.inspect_dataset("invalid/dataset")

    def test_has_image_column_detects_image_feature(self):
        """Verify _has_image_column returns True when Image feature exists."""

        class ImageFeature:
            pass

        features = {"img": ImageFeature(), "label": Mock()}

        result = DatasetInspector._has_image_column(features)

        assert result is True

    def test_has_image_column_returns_false_when_no_image(self):
        """Verify _has_image_column returns False when no Image feature."""
        features = {"text": Mock(), "label": Mock()}

        result = DatasetInspector._has_image_column(features)

        assert result is False

    def test_has_image_column_handles_empty_features(self):
        """Verify _has_image_column handles empty features dict."""
        features = {}

        result = DatasetInspector._has_image_column(features)

        assert result is False

    def test_has_text_column_detects_text_column(self):
        """Verify _has_text_column returns True when text column exists."""
        features = {"text": Mock(), "label": Mock()}

        result = DatasetInspector._has_text_column(features)

        assert result is True

    def test_has_text_column_detects_sentence_column(self):
        """Verify _has_text_column detects 'sentence' column."""
        features = {"sentence": Mock(), "label": Mock()}

        result = DatasetInspector._has_text_column(features)

        assert result is True

    def test_has_text_column_detects_sentence1_and_sentence2(self):
        """Verify _has_text_column detects sentence pair columns."""
        features = {"sentence1": Mock(), "sentence2": Mock(), "label": Mock()}

        result = DatasetInspector._has_text_column(features)

        assert result is True

    def test_has_text_column_detects_question_answer_columns(self):
        """Verify _has_text_column detects QA columns."""
        features = {"question": Mock(), "answer": Mock()}

        result = DatasetInspector._has_text_column(features)

        assert result is True

    def test_has_text_column_detects_premise_hypothesis(self):
        """Verify _has_text_column detects NLI columns."""
        features = {"premise": Mock(), "hypothesis": Mock()}

        result = DatasetInspector._has_text_column(features)

        assert result is True

    def test_has_text_column_detects_document_passage(self):
        """Verify _has_text_column detects document/passage columns."""
        features = {"document": Mock(), "passage": Mock()}

        result = DatasetInspector._has_text_column(features)

        assert result is True

    def test_has_text_column_returns_false_when_no_text(self):
        """Verify _has_text_column returns False when no text columns."""
        features = {"image": Mock(), "label": Mock()}

        result = DatasetInspector._has_text_column(features)

        assert result is False

    def test_has_text_column_handles_empty_features(self):
        """Verify _has_text_column handles empty features dict."""
        features = {}

        result = DatasetInspector._has_text_column(features)

        assert result is False

    def test_get_image_info_finds_image_column(self):
        """Verify _get_image_info finds the image column name."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"img": mock_image_feature, "label": Mock()}

        with patch("src.dataset_loaders.dataset_inspector.load_dataset") as mock_load:
            mock_img = Mock()
            mock_img.mode = "RGB"
            mock_img.size = (32, 32)
            mock_dataset = Mock()
            mock_dataset.__getitem__ = Mock(return_value={"img": mock_img})
            mock_load.return_value = mock_dataset

            col_name, shape = DatasetInspector._get_image_info(
                "unknown/dataset", features
            )

        assert col_name == "img"
        assert shape == (3, 32, 32)

    def test_get_image_info_returns_none_when_no_image(self):
        """Verify _get_image_info returns None when no image column."""
        features = {"text": Mock(), "label": Mock()}

        col_name, shape = DatasetInspector._get_image_info("test/dataset", features)

        assert col_name is None
        assert shape is None

    def test_get_image_info_uses_known_dataset_mnist(self):
        """Verify _get_image_info uses cached shape for MNIST."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"image": mock_image_feature}

        col_name, shape = DatasetInspector._get_image_info("mnist", features)

        assert col_name == "image"
        assert shape == (1, 28, 28)

    def test_get_image_info_uses_known_dataset_femnist(self):
        """Verify _get_image_info uses cached shape for FEMNIST."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"image": mock_image_feature}

        col_name, shape = DatasetInspector._get_image_info("flwrlabs/femnist", features)

        assert col_name == "image"
        assert shape == (1, 28, 28)

    def test_get_image_info_uses_known_dataset_fashion_mnist(self):
        """Verify _get_image_info uses cached shape for Fashion-MNIST."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"image": mock_image_feature}

        col_name, shape = DatasetInspector._get_image_info("fashion_mnist", features)

        assert col_name == "image"
        assert shape == (1, 28, 28)

    def test_get_image_info_uses_known_dataset_cifar10(self):
        """Verify _get_image_info uses cached shape for CIFAR-10."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"img": mock_image_feature}

        col_name, shape = DatasetInspector._get_image_info("cifar10", features)

        assert col_name == "img"
        assert shape == (3, 32, 32)

    def test_get_image_info_uses_known_dataset_cifar100(self):
        """Verify _get_image_info uses cached shape for CIFAR-100."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"img": mock_image_feature}

        col_name, shape = DatasetInspector._get_image_info("cifar100", features)

        assert col_name == "img"
        assert shape == (3, 32, 32)

    @patch("src.dataset_loaders.dataset_inspector.load_dataset")
    def test_get_image_info_loads_sample_for_unknown_dataset(self, mock_load_dataset):
        """Verify _get_image_info loads sample for unknown datasets."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"image": mock_image_feature}

        mock_img = Mock()
        mock_img.mode = "L"
        mock_img.size = (64, 64)
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value={"image": mock_img})
        mock_load_dataset.return_value = mock_dataset

        col_name, shape = DatasetInspector._get_image_info("custom/dataset", features)

        assert col_name == "image"
        assert shape == (1, 64, 64)
        mock_load_dataset.assert_called_once_with("custom/dataset", split="train[:1]")

    @patch("src.dataset_loaders.dataset_inspector.load_dataset")
    def test_get_image_info_detects_rgb_image(self, mock_load_dataset):
        """Verify _get_image_info correctly detects RGB images."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"img": mock_image_feature}

        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_img.size = (224, 224)
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value={"img": mock_img})
        mock_load_dataset.return_value = mock_dataset

        col_name, shape = DatasetInspector._get_image_info("unknown/dataset", features)

        assert shape == (3, 224, 224)

    @patch("src.dataset_loaders.dataset_inspector.load_dataset")
    def test_get_image_info_handles_non_pil_image(self, mock_load_dataset):
        """Verify _get_image_info handles non-PIL image formats."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"image": mock_image_feature}

        mock_img = Mock(spec=[])
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value={"image": mock_img})
        mock_load_dataset.return_value = mock_dataset

        col_name, shape = DatasetInspector._get_image_info("unknown/dataset", features)

        assert shape == (1, 28, 28)

    @patch("src.dataset_loaders.dataset_inspector.load_dataset")
    def test_get_image_info_handles_load_error(self, mock_load_dataset):
        """Verify _get_image_info handles dataset loading errors."""

        class ImageFeature:
            pass

        mock_image_feature = ImageFeature()
        features = {"image": mock_image_feature}

        mock_load_dataset.side_effect = Exception("Network error")

        col_name, shape = DatasetInspector._get_image_info("unknown/dataset", features)

        assert col_name == "image"
        assert shape == (1, 28, 28)

    def test_get_text_columns_extracts_text_column(self):
        """Verify _get_text_columns extracts 'text' column."""
        features = {"text": Mock(), "label": Mock()}

        text_cols = DatasetInspector._get_text_columns(features)

        assert text_cols == ["text"]

    def test_get_text_columns_extracts_sentence_columns(self):
        """Verify _get_text_columns extracts sentence columns."""
        features = {"sentence": Mock(), "label": Mock()}

        text_cols = DatasetInspector._get_text_columns(features)

        assert text_cols == ["sentence"]

    def test_get_text_columns_extracts_sentence_pair_columns(self):
        """Verify _get_text_columns extracts sentence1 and sentence2."""
        features = {"sentence1": Mock(), "sentence2": Mock(), "label": Mock()}

        text_cols = DatasetInspector._get_text_columns(features)

        assert "sentence1" in text_cols
        assert "sentence2" in text_cols

    def test_get_text_columns_extracts_qa_columns(self):
        """Verify _get_text_columns extracts question and answer columns."""
        features = {"question": Mock(), "answer": Mock(), "id": Mock()}

        text_cols = DatasetInspector._get_text_columns(features)

        assert "question" in text_cols
        assert "answer" in text_cols

    def test_get_text_columns_extracts_nli_columns(self):
        """Verify _get_text_columns extracts premise and hypothesis."""
        features = {"premise": Mock(), "hypothesis": Mock(), "label": Mock()}

        text_cols = DatasetInspector._get_text_columns(features)

        assert "premise" in text_cols
        assert "hypothesis" in text_cols

    def test_get_text_columns_extracts_document_columns(self):
        """Verify _get_text_columns extracts document and passage."""
        features = {"document": Mock(), "passage": Mock()}

        text_cols = DatasetInspector._get_text_columns(features)

        assert "document" in text_cols
        assert "passage" in text_cols

    def test_get_text_columns_returns_empty_when_no_text(self):
        """Verify _get_text_columns returns empty list when no text columns."""
        features = {"image": Mock(), "label": Mock()}

        text_cols = DatasetInspector._get_text_columns(features)

        assert text_cols == []

    def test_get_text_columns_handles_empty_features(self):
        """Verify _get_text_columns handles empty features dict."""
        features = {}

        text_cols = DatasetInspector._get_text_columns(features)

        assert text_cols == []

    def test_get_text_columns_preserves_order(self):
        """Verify _get_text_columns preserves expected order."""
        features = {
            "passage": Mock(),
            "text": Mock(),
            "question": Mock(),
            "label": Mock(),
        }

        text_cols = DatasetInspector._get_text_columns(features)

        assert text_cols == ["text", "question", "passage"]
