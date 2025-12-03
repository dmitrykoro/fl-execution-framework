"""
Tests for text visualization utilities in attack snapshots.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from src.attack_utils.snapshot_text_viz import (
    _extract_attack_param,
    _extract_attack_type,
    save_text_samples,
)


class TestExtractAttackParam:
    """Test suite for _extract_attack_param function."""

    def test_should_extract_parameter_from_single_dict(self) -> None:
        """Test that parameter is extracted from single dict config."""
        config = {"attack_type": "gaussian_noise", "target_noise_snr": 10}
        result = _extract_attack_param(config, "target_noise_snr")
        assert result == 10

    def test_should_extract_parameter_from_list_of_dicts(self) -> None:
        """Test that parameter is extracted from first dict in list."""
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise", "target_noise_snr": 15},
        ]
        result = _extract_attack_param(config, "target_noise_snr")
        assert result == "?"

    def test_should_return_default_when_parameter_not_found(self) -> None:
        """Test that default value is returned when parameter not found."""
        config = {"attack_type": "label_flipping"}
        result = _extract_attack_param(config, "missing_param", default="default_val")
        assert result == "default_val"

    def test_should_return_question_mark_default_when_parameter_not_found(self) -> None:
        """Test that '?' is returned as default when parameter not found."""
        config = {"attack_type": "label_flipping"}
        result = _extract_attack_param(config, "missing_param")
        assert result == "?"

    def test_should_search_multiple_parameter_names(self) -> None:
        """Test that first matching parameter name is returned."""
        config = {"type": "gaussian_noise", "snr": 20}
        result = _extract_attack_param(config, "attack_type", "type")
        assert result == "gaussian_noise"

    def test_should_handle_empty_list_config(self) -> None:
        """Test that default is returned for empty list config."""
        config = []
        result = _extract_attack_param(config, "any_param", default="empty")
        assert result == "empty"

    @pytest.mark.parametrize(
        "config,params,expected",
        [
            ({"a": 1, "b": 2}, ("a",), 1),
            ({"a": 1, "b": 2}, ("b",), 2),
            ({"a": 1, "b": 2}, ("c", "a"), 1),
            ([{"a": 1}], ("a",), 1),
            (None, ("a",), "?"),
        ],
    )
    def test_parameter_extraction_variations(
        self, config: Any, params: tuple, expected: Any
    ) -> None:
        """Test various parameter extraction scenarios."""
        result = _extract_attack_param(config, *params)
        assert result == expected


class TestExtractAttackType:
    """Test suite for _extract_attack_type function."""

    def test_should_extract_attack_type_from_single_dict(self) -> None:
        """Test that attack_type is extracted from single dict config."""
        config = {"attack_type": "label_flipping"}
        result = _extract_attack_type(config)
        assert result == "label_flipping"

    def test_should_extract_type_field_as_fallback(self) -> None:
        """Test that 'type' field is used when 'attack_type' not present."""
        config = {"type": "gaussian_noise"}
        result = _extract_attack_type(config)
        assert result == "gaussian_noise"

    def test_should_join_multiple_attack_types_from_list(self) -> None:
        """Test that multiple attack types are joined with underscore."""
        config = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise"},
        ]
        result = _extract_attack_type(config)
        assert result == "label_flipping_gaussian_noise"

    def test_should_return_unknown_for_empty_list(self) -> None:
        """Test that 'unknown' is returned for empty list config."""
        config = []
        result = _extract_attack_type(config)
        assert result == "unknown"

    def test_should_return_unknown_when_attack_type_missing(self) -> None:
        """Test that 'unknown' is returned when attack_type missing."""
        config = {"other_field": "value"}
        result = _extract_attack_type(config)
        assert result == "unknown"

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({"attack_type": "label_flipping"}, "label_flipping"),
            ({"attack_type": "gaussian_noise"}, "gaussian_noise"),
            ({"attack_type": "token_replacement"}, "token_replacement"),
            ([{"attack_type": "label_flipping"}], "label_flipping"),
            (
                [{"attack_type": "label_flipping"}, {"attack_type": "gaussian_noise"}],
                "label_flipping_gaussian_noise",
            ),
            (
                [
                    {"attack_type": "label_flipping"},
                    {"attack_type": "gaussian_noise"},
                    {"attack_type": "token_replacement"},
                ],
                "label_flipping_gaussian_noise_token_replacement",
            ),
            ([], "unknown"),
            ({}, "unknown"),
        ],
    )
    def test_attack_type_extraction_variations(
        self, config: Any, expected: str
    ) -> None:
        """Test various attack type extraction scenarios."""
        result = _extract_attack_type(config)
        assert result == expected


class TestSaveTextSamples:
    """Test suite for save_text_samples function."""

    @pytest.fixture
    def sample_labels(self) -> tuple:
        """Create sample labels for testing."""
        labels = np.array([1, 0, 1])
        original_labels = np.array([0, 0, 1])
        return labels, original_labels

    @pytest.fixture
    def sample_token_ids(self) -> tuple:
        """Create sample token IDs for testing."""
        original = np.array([[101, 2023, 2003, 102], [101, 1045, 2572, 102]])
        poisoned = np.array([[101, 9999, 2003, 102], [101, 1045, 2572, 102]])
        return original, poisoned

    @pytest.fixture
    def mock_tokenizer(self) -> Mock:
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens: " ".join(
            [f"token_{id}" for id in ids if id not in [101, 102]]
        )
        return tokenizer

    def test_should_save_labels_only_when_no_tokenizer(
        self, tmp_path: Path, sample_labels: tuple
    ) -> None:
        """Test that only labels are saved when tokenizer not provided."""
        labels, original_labels = sample_labels
        filepath = tmp_path / "test_output.txt"

        save_text_samples(
            labels=labels, original_labels=original_labels, filepath=filepath
        )

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "Sample Labels (Poisoned vs Original)" in content
        assert "Sample 0: 1 (was 0)" in content
        assert "Sample 1: 0 (was 0)" in content
        assert "Sample 2: 1 (was 1)" in content

    def test_should_save_token_replacement_details_with_tokenizer(
        self, tmp_path: Path, sample_token_ids: tuple, mock_tokenizer: Mock
    ) -> None:
        """Test that token replacement details are saved with tokenizer."""
        original_ids, poisoned_ids = sample_token_ids
        labels = np.array([1, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "test_output.txt"
        attack_config = {
            "attack_type": "token_replacement",
            "target_vocabulary": "negative",
            "replacement_strategy": "random",
            "replacement_probability": 0.5,
        }

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            attack_config=attack_config,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "TOKEN REPLACEMENT ATTACK VISUALIZATION" in content
        assert "Attack Type: token_replacement" in content
        assert "Target Vocabulary: negative" in content
        assert "Replacement Strategy: random" in content
        assert "Replacement Probability: 0.5" in content
        assert "--- Sample 0 ---" in content
        assert "ORIGINAL:" in content
        assert "POISONED:" in content

    def test_should_skip_unchanged_samples(
        self, tmp_path: Path, mock_tokenizer: Mock
    ) -> None:
        """Test that unchanged samples are skipped in output."""
        original_ids = np.array([[101, 2023, 102], [101, 1045, 102]])
        poisoned_ids = np.array([[101, 2023, 102], [101, 1045, 102]])
        labels = np.array([0, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "test_output.txt"

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "2 samples unchanged (skipped)" in content
        assert "0 samples modified" in content

    def test_should_show_replacement_statistics(
        self, tmp_path: Path, sample_token_ids: tuple, mock_tokenizer: Mock
    ) -> None:
        """Test that replacement statistics are displayed."""
        original_ids, poisoned_ids = sample_token_ids
        labels = np.array([1, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "test_output.txt"

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        content = filepath.read_text(encoding="utf-8")
        assert "tokens replaced" in content
        assert "[REPLACED]" in content

    def test_should_show_label_changes(
        self, tmp_path: Path, sample_token_ids: tuple, mock_tokenizer: Mock
    ) -> None:
        """Test that label changes are displayed."""
        original_ids, poisoned_ids = sample_token_ids
        labels = np.array([1, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "test_output.txt"

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        content = filepath.read_text(encoding="utf-8")
        assert "Label: 0 → 1 (FLIPPED)" in content or "Label: 0 (unchanged)" in content

    def test_should_handle_sequence_labels(
        self, tmp_path: Path, sample_token_ids: tuple, mock_tokenizer: Mock
    ) -> None:
        """Test handling of sequence labels (masked language modeling)."""
        original_ids, poisoned_ids = sample_token_ids
        labels = np.array([[-100, 1, -100, -100], [-100, -100, 2, -100]])
        original_labels = np.array([[-100, 0, -100, -100], [-100, -100, 2, -100]])
        filepath = tmp_path / "test_output.txt"

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        content = filepath.read_text(encoding="utf-8")
        assert "masked tokens" in content or "Label:" in content

    def test_should_handle_decoding_errors(
        self, tmp_path: Path, sample_token_ids: tuple
    ) -> None:
        """Test that decoding errors are handled gracefully."""
        original_ids, poisoned_ids = sample_token_ids
        labels = np.array([1, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "test_output.txt"

        mock_tokenizer = Mock()
        mock_tokenizer.decode.side_effect = Exception("Decoding failed")

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "[Decoding error:" in content
        assert "Original token IDs" in content
        assert "Poisoned token IDs" in content

    def test_should_handle_list_attack_config(
        self, tmp_path: Path, sample_token_ids: tuple, mock_tokenizer: Mock
    ) -> None:
        """Test handling of list-style attack config."""
        original_ids, poisoned_ids = sample_token_ids
        labels = np.array([1, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "test_output.txt"
        attack_config = [
            {"attack_type": "token_replacement", "target_vocabulary": "negative"}
        ]

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            attack_config=attack_config,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "TOKEN REPLACEMENT ATTACK VISUALIZATION" in content

    @pytest.mark.parametrize("num_samples", [1, 5, 10])
    def test_should_handle_various_sample_counts(
        self, tmp_path: Path, num_samples: int
    ) -> None:
        """Test that various sample counts are handled correctly."""
        labels = np.random.randint(0, 10, size=num_samples)
        original_labels = np.random.randint(0, 10, size=num_samples)
        filepath = tmp_path / "test_output.txt"

        save_text_samples(
            labels=labels, original_labels=original_labels, filepath=filepath
        )

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        for i in range(num_samples):
            assert f"Sample {i}:" in content

    def test_should_handle_unicode_text(
        self, tmp_path: Path, sample_token_ids: tuple
    ) -> None:
        """Test that unicode text is handled correctly."""
        original_ids, poisoned_ids = sample_token_ids
        labels = np.array([1, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "test_output.txt"

        mock_tokenizer = Mock()
        call_count = [0]

        def decode_side_effect(ids, skip_special_tokens):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return "Hello 世界 emoji😊"
            else:
                return "Different text with unicode 日本語"

        mock_tokenizer.decode.side_effect = decode_side_effect

        save_text_samples(
            labels=labels,
            original_labels=original_labels,
            filepath=filepath,
            tokenizer=mock_tokenizer,
            input_ids_original=original_ids,
            input_ids_poisoned=poisoned_ids,
        )

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "世界" in content or "日本語" in content or "emoji" in content

    def test_should_create_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created if needed."""
        labels = np.array([1, 0])
        original_labels = np.array([0, 0])
        filepath = tmp_path / "subdir" / "nested" / "test_output.txt"

        filepath.parent.mkdir(parents=True, exist_ok=True)
        save_text_samples(
            labels=labels, original_labels=original_labels, filepath=filepath
        )

        assert filepath.exists()
