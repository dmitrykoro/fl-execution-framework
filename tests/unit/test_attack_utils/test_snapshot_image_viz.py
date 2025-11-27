"""
Tests for image visualization utilities in attack snapshots.
"""

import math
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.attack_utils.snapshot_image_viz import (
    _build_attack_title,
    _build_single_attack_title,
    _display_image,
    _extract_attack_param,
    _extract_attack_type,
    _normalize_axes,
    save_image_grid,
)


class TestDisplayImage:
    """Test suite for _display_image function."""

    @pytest.fixture
    def mock_axes(self) -> Mock:
        """Create mock matplotlib axes."""
        ax = Mock()
        ax.imshow = Mock()
        return ax

    def test_should_display_grayscale_image(self, mock_axes: Mock) -> None:
        """Test that grayscale image is displayed with gray colormap."""
        image = np.random.rand(1, 28, 28)
        _display_image(mock_axes, image)

        mock_axes.imshow.assert_called_once()
        call_args = mock_axes.imshow.call_args
        assert call_args[1]["cmap"] == "gray"
        assert call_args[0][0].shape == (28, 28)

    def test_should_display_rgb_image(self, mock_axes: Mock) -> None:
        """Test that RGB image is displayed with transposed axes."""
        image = np.random.rand(3, 32, 32)
        _display_image(mock_axes, image)

        mock_axes.imshow.assert_called_once()
        call_args = mock_axes.imshow.call_args
        displayed_image = call_args[0][0]
        assert displayed_image.shape == (32, 32, 3)

    @pytest.mark.parametrize(
        "shape,expected_cmap",
        [
            ((1, 28, 28), "gray"),
            ((3, 32, 32), None),
            ((1, 64, 64), "gray"),
        ],
    )
    def test_display_image_variations(
        self, mock_axes: Mock, shape: tuple, expected_cmap: str
    ) -> None:
        """Test various image shapes are handled correctly."""
        image = np.random.rand(*shape)
        _display_image(mock_axes, image)

        mock_axes.imshow.assert_called_once()
        if expected_cmap:
            call_args = mock_axes.imshow.call_args
            assert call_args[1].get("cmap") == expected_cmap


class TestNormalizeAxes:
    """Test suite for _normalize_axes function."""

    def test_should_normalize_single_axis(self) -> None:
        """Test that single axis is wrapped in nested list."""
        ax = Mock()
        result = _normalize_axes(ax, rows=1, cols=1)

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] == ax

    def test_should_normalize_single_row(self) -> None:
        """Test that single row is wrapped in list."""
        axes = [Mock(), Mock(), Mock()]
        result = _normalize_axes(axes, rows=1, cols=3)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == axes

    def test_should_normalize_single_column(self) -> None:
        """Test that single column is converted to nested list."""
        axes = [Mock(), Mock(), Mock()]
        result = _normalize_axes(axes, rows=3, cols=1)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(row) == 1 for row in result)
        assert result[0][0] == axes[0]
        assert result[1][0] == axes[1]
        assert result[2][0] == axes[2]

    def test_should_normalize_grid(self) -> None:
        """Test that 2D grid is returned as-is."""
        axes = [[Mock(), Mock()], [Mock(), Mock()]]
        result = _normalize_axes(axes, rows=2, cols=2)

        assert result == axes

    @pytest.mark.parametrize(
        "rows,cols,axes_input",
        [
            (1, 1, Mock()),
            (1, 3, [Mock(), Mock(), Mock()]),
            (3, 1, [Mock(), Mock(), Mock()]),
            (2, 2, [[Mock(), Mock()], [Mock(), Mock()]]),
        ],
    )
    def test_normalize_axes_variations(
        self, rows: int, cols: int, axes_input: Any
    ) -> None:
        """Test various axes configurations are normalized correctly."""
        result = _normalize_axes(axes_input, rows=rows, cols=cols)

        assert isinstance(result, list)
        if rows == 1 and cols == 1:
            assert len(result) == 1
            assert len(result[0]) == 1
        elif rows == 1:
            assert len(result) == 1
            assert len(result[0]) == cols
        elif cols == 1:
            assert len(result) == rows
            assert all(len(row) == 1 for row in result)
        else:
            assert len(result) == rows
            assert all(len(row) == cols for row in result)


class TestExtractAttackParam:
    """Test suite for _extract_attack_param function."""

    def test_should_extract_parameter_from_dict(self) -> None:
        """Test that parameter is extracted from dict config."""
        config = {"attack_type": "gaussian_noise", "target_noise_snr": 10}
        result = _extract_attack_param(config, "target_noise_snr")
        assert result == 10

    def test_should_extract_from_first_dict_in_list(self) -> None:
        """Test that parameter is extracted from first dict in list."""
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise"},
        ]
        result = _extract_attack_param(config, "attack_type")
        assert result == "label_flipping"

    def test_should_return_default_when_not_found(self) -> None:
        """Test that default value is returned when parameter not found."""
        config = {"attack_type": "label_flipping"}
        result = _extract_attack_param(config, "missing_param", default="default")
        assert result == "default"

    @pytest.mark.parametrize(
        "config,param,default,expected",
        [
            ({"a": 1}, "a", "?", 1),
            ({"a": 1}, "b", "?", "?"),
            ([{"a": 1}], "a", "?", 1),
            ([], "a", "?", "?"),
            (None, "a", "?", "?"),
        ],
    )
    def test_extract_param_variations(
        self, config: Any, param: str, default: Any, expected: Any
    ) -> None:
        """Test various parameter extraction scenarios."""
        result = _extract_attack_param(config, param, default=default)
        assert result == expected


class TestExtractAttackType:
    """Test suite for _extract_attack_type function."""

    def test_should_extract_attack_type_from_dict(self) -> None:
        """Test that attack_type is extracted from dict."""
        config = {"attack_type": "label_flipping"}
        result = _extract_attack_type(config)
        assert result == "label_flipping"

    def test_should_use_type_field_as_fallback(self) -> None:
        """Test that 'type' field is used when 'attack_type' not present."""
        config = {"type": "gaussian_noise"}
        result = _extract_attack_type(config)
        assert result == "gaussian_noise"

    def test_should_join_multiple_attack_types(self) -> None:
        """Test that multiple attack types are joined with underscore."""
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise"},
        ]
        result = _extract_attack_type(config)
        assert result == "label_flipping_gaussian_noise"

    def test_should_return_unknown_for_empty_list(self) -> None:
        """Test that 'unknown' is returned for empty list."""
        config = []
        result = _extract_attack_type(config)
        assert result == "unknown"

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({"attack_type": "label_flipping"}, "label_flipping"),
            ({"attack_type": "gaussian_noise"}, "gaussian_noise"),
            ({"attack_type": "token_replacement"}, "token_replacement"),
            ({"type": "label_flipping"}, "label_flipping"),
            (
                [{"attack_type": "label_flipping"}, {"attack_type": "gaussian_noise"}],
                "label_flipping_gaussian_noise",
            ),
            ([], "unknown"),
            ({}, "unknown"),
        ],
    )
    def test_extract_attack_type_variations(self, config: Any, expected: str) -> None:
        """Test various attack type extraction scenarios."""
        result = _extract_attack_type(config)
        assert result == expected


class TestBuildSingleAttackTitle:
    """Test suite for _build_single_attack_title function."""

    @pytest.fixture
    def sample_labels(self) -> tuple:
        """Create sample labels for testing."""
        labels = np.array([5, 3, 7])
        original_labels = np.array([1, 3, 2])
        return labels, original_labels

    def test_should_build_label_flipping_title_side_by_side(
        self, sample_labels: tuple
    ) -> None:
        """Test label flipping title for side-by-side style."""
        labels, original_labels = sample_labels
        config = {"attack_type": "label_flipping"}
        attack_type = "label_flipping"

        title = _build_single_attack_title(
            config, attack_type, labels, original_labels, index=0, style="side_by_side"
        )

        assert "Poisoned" in title
        assert "Label: 5" in title

    def test_should_build_label_flipping_title_fallback(
        self, sample_labels: tuple
    ) -> None:
        """Test label flipping title for fallback style."""
        labels, original_labels = sample_labels
        config = {"attack_type": "label_flipping"}
        attack_type = "label_flipping"

        title = _build_single_attack_title(
            config, attack_type, labels, original_labels, index=0, style="fallback"
        )

        assert "Label: 5" in title
        assert "(was 1)" in title

    def test_should_build_gaussian_noise_title_side_by_side(
        self, sample_labels: tuple
    ) -> None:
        """Test gaussian noise title for side-by-side style."""
        labels, original_labels = sample_labels
        config = {"attack_type": "gaussian_noise", "target_noise_snr": 10}
        attack_type = "gaussian_noise"

        title = _build_single_attack_title(
            config, attack_type, labels, original_labels, index=0, style="side_by_side"
        )

        assert "Poisoned (Noise)" in title
        assert "SNR: 10dB" in title
        assert "Label: 5" in title

    def test_should_build_gaussian_noise_title_fallback(
        self, sample_labels: tuple
    ) -> None:
        """Test gaussian noise title for fallback style."""
        labels, original_labels = sample_labels
        config = {"attack_type": "gaussian_noise", "target_noise_snr": 15}
        attack_type = "gaussian_noise"

        title = _build_single_attack_title(
            config, attack_type, labels, original_labels, index=1, style="fallback"
        )

        assert "Noisy (SNR: 15dB)" in title
        assert "Label: 3" in title

    def test_should_build_token_replacement_title(self, sample_labels: tuple) -> None:
        """Test token replacement title."""
        labels, original_labels = sample_labels
        config = {"attack_type": "token_replacement"}
        attack_type = "token_replacement"

        title = _build_single_attack_title(
            config, attack_type, labels, original_labels, index=0, style="side_by_side"
        )

        assert "Token poisoned" in title
        assert "Label: 5" in title

    def test_should_build_unknown_attack_title(self, sample_labels: tuple) -> None:
        """Test unknown attack type title."""
        labels, original_labels = sample_labels
        config = {"attack_type": "custom_attack"}
        attack_type = "custom_attack"

        title = _build_single_attack_title(
            config, attack_type, labels, original_labels, index=0, style="side_by_side"
        )

        assert "Poisoned (custom_attack)" in title
        assert "Label: 5" in title

    @pytest.mark.parametrize(
        "attack_type,style,should_contain",
        [
            ("label_flipping", "side_by_side", ["Poisoned", "Label:"]),
            ("label_flipping", "fallback", ["Label:", "(was"]),
            ("gaussian_noise", "side_by_side", ["Poisoned", "SNR:", "Label:"]),
            ("gaussian_noise", "fallback", ["Noisy", "SNR:", "Label:"]),
            ("token_replacement", "side_by_side", ["Token poisoned", "Label:"]),
            ("custom", "side_by_side", ["Poisoned (custom)", "Label:"]),
        ],
    )
    def test_build_title_variations(
        self, sample_labels: tuple, attack_type: str, style: str, should_contain: list
    ) -> None:
        """Test various title building scenarios."""
        labels, original_labels = sample_labels
        config = {"attack_type": attack_type, "target_noise_snr": 10}

        title = _build_single_attack_title(
            config, attack_type, labels, original_labels, index=0, style=style
        )

        for expected in should_contain:
            assert expected in title


class TestBuildAttackTitle:
    """Test suite for _build_attack_title function."""

    @pytest.fixture
    def sample_labels(self) -> tuple:
        """Create sample labels for testing."""
        labels = np.array([5, 3, 7])
        original_labels = np.array([1, 3, 2])
        return labels, original_labels

    def test_should_build_single_attack_title(self, sample_labels: tuple) -> None:
        """Test that single attack uses _build_single_attack_title."""
        labels, original_labels = sample_labels
        config = {"attack_type": "label_flipping"}

        title = _build_attack_title(
            config, "label_flipping", labels, original_labels, 0, "side_by_side"
        )

        assert "Poisoned" in title
        assert "Label: 5" in title

    def test_should_build_composite_attack_title_side_by_side(
        self, sample_labels: tuple
    ) -> None:
        """Test composite attack title for side-by-side style."""
        labels, original_labels = sample_labels
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise", "target_noise_snr": 10},
        ]

        title = _build_attack_title(
            config,
            "label_flipping_gaussian_noise",
            labels,
            original_labels,
            0,
            "side_by_side",
        )

        assert "Poisoned" in title
        assert "Label: 5" in title or "Noise:" in title

    def test_should_build_composite_attack_title_fallback(
        self, sample_labels: tuple
    ) -> None:
        """Test composite attack title for fallback style."""
        labels, original_labels = sample_labels
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise", "target_noise_snr": 10},
        ]

        title = _build_attack_title(
            config,
            "label_flipping_gaussian_noise",
            labels,
            original_labels,
            0,
            "fallback",
        )

        assert "Label Flip:" in title or "Noise" in title

    def test_should_handle_single_item_list_config(self, sample_labels: tuple) -> None:
        """Test that single-item list config is handled correctly."""
        labels, original_labels = sample_labels
        config = [{"attack_type": "label_flipping"}]

        title = _build_attack_title(
            config, "label_flipping", labels, original_labels, 0, "side_by_side"
        )

        assert "Poisoned" in title or "Label:" in title

    @pytest.mark.parametrize("style", ["side_by_side", "fallback"])
    def test_build_composite_title_styles(
        self, sample_labels: tuple, style: str
    ) -> None:
        """Test composite titles with different styles."""
        labels, original_labels = sample_labels
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise", "target_noise_snr": 15},
        ]

        title = _build_attack_title(
            config, "label_flipping_gaussian_noise", labels, original_labels, 0, style
        )

        assert isinstance(title, str)
        assert len(title) > 0


class TestSaveImageGrid:
    """Test suite for save_image_grid function."""

    @pytest.fixture
    def sample_images(self) -> tuple:
        """Create sample images for testing."""
        num_samples = 8
        images = np.random.rand(num_samples, 3, 32, 32)
        labels = np.random.randint(0, 10, size=num_samples)
        original_labels = np.random.randint(0, 10, size=num_samples)
        return images, labels, original_labels

    @pytest.fixture
    def grayscale_images(self) -> tuple:
        """Create grayscale sample images for testing."""
        num_samples = 4
        images = np.random.rand(num_samples, 1, 28, 28)
        labels = np.random.randint(0, 10, size=num_samples)
        original_labels = np.random.randint(0, 10, size=num_samples)
        return images, labels, original_labels

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_should_save_grid_without_originals(
        self,
        mock_subplots: Mock,
        mock_savefig: Mock,
        mock_close: Mock,
        tmp_path: Path,
        sample_images: tuple,
    ) -> None:
        """Test that grid is saved without original images."""
        images, labels, original_labels = sample_images
        filepath = tmp_path / "test_grid.png"
        config = {"attack_type": "label_flipping"}

        mock_fig = Mock()
        mock_axes = [Mock() for _ in range(8)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        save_image_grid(images, labels, original_labels, filepath, config)

        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once_with(filepath, dpi=150, bbox_inches="tight")
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.Line2D")
    def test_should_save_grid_with_originals(
        self,
        mock_line2d: Mock,
        mock_subplots: Mock,
        mock_savefig: Mock,
        mock_close: Mock,
        tmp_path: Path,
        sample_images: tuple,
    ) -> None:
        """Test that grid is saved with original images for comparison."""
        images, labels, original_labels = sample_images
        original_images = np.random.rand(*images.shape)
        filepath = tmp_path / "test_grid.png"
        config = {"attack_type": "label_flipping"}

        num_samples = len(images)
        pairs_per_row = 4
        cols = pairs_per_row * 2
        rows = math.ceil(num_samples / pairs_per_row)

        mock_fig = Mock()
        mock_axes = [[Mock() for _ in range(cols)] for _ in range(rows)]

        for row in mock_axes:
            for ax in row:
                mock_bbox = Mock()
                mock_bbox.x0 = 0.1
                mock_bbox.x1 = 0.2
                mock_bbox.y0 = 0.3
                mock_bbox.y1 = 0.4
                ax.get_position.return_value = mock_bbox

        mock_subplots.return_value = (mock_fig, mock_axes)

        save_image_grid(
            images, labels, original_labels, filepath, config, original_images
        )

        mock_subplots.assert_called_once()
        call_args = mock_subplots.call_args
        assert call_args[0][0] == rows
        assert call_args[0][1] == cols
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_should_handle_grayscale_images(
        self,
        mock_subplots: Mock,
        mock_savefig: Mock,
        mock_close: Mock,
        tmp_path: Path,
        grayscale_images: tuple,
    ) -> None:
        """Test that grayscale images are handled correctly."""
        images, labels, original_labels = grayscale_images
        filepath = tmp_path / "test_grid.png"
        config = {"attack_type": "label_flipping"}

        mock_fig = Mock()
        mock_axes = [Mock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        save_image_grid(images, labels, original_labels, filepath, config)

        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    @pytest.mark.parametrize("num_samples", [1, 4, 8, 16])
    def test_should_handle_various_grid_sizes(
        self,
        mock_subplots: Mock,
        mock_savefig: Mock,
        mock_close: Mock,
        tmp_path: Path,
        num_samples: int,
    ) -> None:
        """Test that various grid sizes are handled correctly."""
        images = np.random.rand(num_samples, 3, 32, 32)
        labels = np.random.randint(0, 10, size=num_samples)
        original_labels = np.random.randint(0, 10, size=num_samples)
        filepath = tmp_path / "test_grid.png"
        config = {"attack_type": "label_flipping"}

        max_cols = 8
        if num_samples <= max_cols:
            expected_rows, expected_cols = 1, num_samples
        else:
            expected_cols = max_cols
            expected_rows = math.ceil(num_samples / max_cols)

        mock_fig = Mock()
        if num_samples == 1:
            mock_axes = Mock()
        else:
            mock_axes = [Mock() for _ in range(expected_rows * expected_cols)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        save_image_grid(images, labels, original_labels, filepath, config)

        mock_subplots.assert_called_once()
        call_args = mock_subplots.call_args
        assert call_args[0][0] == expected_rows
        assert call_args[0][1] == expected_cols
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    @pytest.mark.parametrize(
        "attack_config",
        [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise", "target_noise_snr": 10},
            {"attack_type": "token_replacement"},
            [
                {"attack_type": "label_flipping"},
                {"attack_type": "gaussian_noise", "target_noise_snr": 10},
            ],
        ],
    )
    def test_should_handle_various_attack_configs(
        self,
        mock_subplots: Mock,
        mock_savefig: Mock,
        mock_close: Mock,
        tmp_path: Path,
        attack_config: Any,
    ) -> None:
        """Test that various attack configurations are handled."""
        images = np.random.rand(4, 3, 32, 32)
        labels = np.random.randint(0, 10, size=4)
        original_labels = np.random.randint(0, 10, size=4)
        filepath = tmp_path / "test_grid.png"

        mock_fig = Mock()
        mock_axes = [Mock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        save_image_grid(images, labels, original_labels, filepath, attack_config)

        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_should_handle_list_attack_config(
        self,
        mock_subplots: Mock,
        mock_savefig: Mock,
        mock_close: Mock,
        tmp_path: Path,
        sample_images: tuple,
    ) -> None:
        """Test that list-style attack config is handled."""
        images, labels, original_labels = sample_images
        filepath = tmp_path / "test_grid.png"
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise", "target_noise_snr": 10},
        ]

        mock_fig = Mock()
        mock_axes = [Mock() for _ in range(8)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        save_image_grid(images, labels, original_labels, filepath, config)

        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_should_use_correct_figsize(
        self,
        mock_subplots: Mock,
        mock_savefig: Mock,
        mock_close: Mock,
        tmp_path: Path,
        sample_images: tuple,
    ) -> None:
        """Test that correct figsize is calculated."""
        images, labels, original_labels = sample_images
        filepath = tmp_path / "test_grid.png"
        config = {"attack_type": "label_flipping"}

        mock_fig = Mock()
        mock_axes = [Mock() for _ in range(8)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        save_image_grid(images, labels, original_labels, filepath, config)

        call_args = mock_subplots.call_args
        figsize = call_args[1]["figsize"]
        assert isinstance(figsize, tuple)
        assert len(figsize) == 2
        assert figsize[0] > 0
        assert figsize[1] > 0
