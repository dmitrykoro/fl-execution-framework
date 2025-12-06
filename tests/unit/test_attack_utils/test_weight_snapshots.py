"""Unit tests for weight snapshot utilities."""

import json
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from src.attack_utils.weight_snapshots import (
    _compute_weight_diff_statistics,
    _compute_weight_statistics,
    _get_weight_snapshot_dir,
    list_weight_snapshots,
    load_weight_snapshot,
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


class TestWeightStatistics:
    """Tests for statistical computation functions."""

    def test_compute_weight_statistics(self, mock_params):
        """Tests basic statistics calculation."""
        stats = _compute_weight_statistics(mock_params)

        assert "mean" in stats
        assert "std" in stats
        assert stats["num_parameters"] == 6
        assert stats["num_layers"] == 2
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0

    def test_compute_weight_diff_statistics(self, mock_params, mock_params_after):
        """Tests difference statistics calculation."""
        stats = _compute_weight_diff_statistics(mock_params, mock_params_after)

        assert "diff_mean" in stats
        assert "diff_l2_norm" in stats
        assert stats["num_changed"] == 6
        assert np.isclose(stats["diff_mean"], 0.1)


class TestSnapshotStorage:
    """Tests for saving and loading snapshots."""

    def test_get_weight_snapshot_dir(self, tmp_path):
        """Tests directory path generation."""
        output_dir = tmp_path / "out"
        client_id = 1
        round_num = 5

        path = _get_weight_snapshot_dir(str(output_dir), client_id, round_num)

        expected = (
            output_dir
            / "weight_snapshots_0"
            / f"client_{client_id}"
            / f"round_{round_num}"
        )
        assert path == expected
        assert path.exists()

    def test_save_weight_snapshot(self, tmp_path, mock_params, mock_params_after):
        """Tests the full save process."""
        output_dir = tmp_path / "out"
        client_id = 99
        round_num = 1
        attack_type = "test_attack"
        config = {"param": "value"}

        save_weight_snapshot(
            parameters_before=mock_params,
            parameters_after=mock_params_after,
            attack_type=attack_type,
            attack_config=config,
            client_id=client_id,
            round_num=round_num,
            output_dir=str(output_dir),
            save_histogram=False,
        )

        snapshot_dir = _get_weight_snapshot_dir(str(output_dir), client_id, round_num)
        json_path = snapshot_dir / f"{attack_type}_weight_metadata.json"

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert data["client_id"] == client_id
        assert data["attack_type"] == attack_type
        assert data["statistics"]["before"]["num_parameters"] == 6

    def test_save_histogram_handling(self, tmp_path, mock_params, mock_params_after):
        """Tests that histogram saving is attempted."""
        with patch(
            "src.attack_utils.weight_snapshots._save_weight_histogram"
        ) as mock_hist:
            save_weight_snapshot(
                parameters_before=mock_params,
                parameters_after=mock_params_after,
                attack_type="hist_test",
                attack_config={},
                client_id=1,
                round_num=1,
                output_dir=str(tmp_path),
                save_histogram=True,
            )
            mock_hist.assert_called_once()

    def test_load_weight_snapshot(self, tmp_path):
        """Tests loading an existing snapshot."""
        fpath = tmp_path / "test.json"
        data = {"test": "data"}
        with open(fpath, "w") as f:
            json.dump(data, f)

        loaded = load_weight_snapshot(str(fpath))
        assert loaded == data

    def test_load_missing_snapshot(self, tmp_path):
        """Tests loading a non-existent file."""
        assert load_weight_snapshot(str(tmp_path / "ghost.json")) is None


class TestListSnapshots:
    """Tests for listing snapshots."""

    def test_list_weight_snapshots(self, tmp_path):
        """Tests listing multiple snapshots."""
        output_dir = tmp_path / "out"
        strategy_num = 0

        base = output_dir / f"weight_snapshots_{strategy_num}"
        (base / "client_1" / "round_1").mkdir(parents=True)
        (base / "client_1" / "round_2").mkdir(parents=True)

        (base / "client_1" / "round_1" / "a_weight_metadata.json").touch()
        (base / "client_1" / "round_2" / "b_weight_metadata.json").touch()

        (base / "client_1" / "round_1" / "image.png").touch()

        snapshots = list_weight_snapshots(str(output_dir), strategy_num)
        assert len(snapshots) == 2
        assert snapshots[0].name == "a_weight_metadata.json"
        assert snapshots[1].name == "b_weight_metadata.json"
