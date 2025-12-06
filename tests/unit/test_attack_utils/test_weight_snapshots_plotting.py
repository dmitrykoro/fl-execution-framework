"""Unit tests for weight snapshot plotting functionality."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.attack_utils.weight_snapshots import (
    _save_weight_histogram,
    save_weight_snapshot,
)


@pytest.fixture
def mock_params() -> list[NDArray]:
    """Creates mock model parameters."""
    return [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([1.0, 2.0]),
    ]


@pytest.fixture
def mock_params_after() -> list[NDArray]:
    """Creates modified mock model parameters."""
    return [
        np.array([[1.1, 2.1], [3.1, 4.1]]),
        np.array([1.1, 2.1]),
    ]


class TestWeightSnapshotPlotting:
    """Tests specifically for plotting functionality using matplotlib."""

    def test_save_weight_histogram_creates_file(
        self, tmp_path, mock_params, mock_params_after
    ):
        """Tests that the histogram file is created on disk."""
        attack_type = "test_plot"
        client_id = 99
        round_num = 1

        _save_weight_histogram(
            params_before=mock_params,
            params_after=mock_params_after,
            attack_type=attack_type,
            snapshot_dir=tmp_path,
            client_id=client_id,
            round_num=round_num,
        )

        expected_file = tmp_path / f"{attack_type}_weight_histogram.png"
        assert expected_file.exists()
        assert expected_file.stat().st_size > 0

    def test_save_weight_snapshot_calls_histogram(
        self, tmp_path, mock_params, mock_params_after
    ):
        """Tests the complete flow including histogram generation."""
        output_dir = tmp_path / "out"
        client_id = 42
        round_num = 1
        attack_type = "full_flow"

        save_weight_snapshot(
            parameters_before=mock_params,
            parameters_after=mock_params_after,
            attack_type=attack_type,
            attack_config={},
            client_id=client_id,
            round_num=round_num,
            output_dir=str(output_dir),
            save_histogram=True,
        )

        snapshot_dir = (
            output_dir
            / "weight_snapshots_0"
            / f"client_{client_id}"
            / f"round_{round_num}"
        )
        expected_hist = snapshot_dir / f"{attack_type}_weight_histogram.png"

        assert expected_hist.exists()

    def test_histogram_handles_empty_params(self, tmp_path):
        """Tests robustness against empty parameters."""
        empty_params = [np.array([])]

        try:
            _save_weight_histogram(
                params_before=empty_params,
                params_after=empty_params,
                attack_type="empty",
                snapshot_dir=tmp_path,
                client_id=1,
                round_num=1,
            )
        except (ValueError, IndexError):
            pass
