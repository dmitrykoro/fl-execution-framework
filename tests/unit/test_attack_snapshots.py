"""Unit tests for attack snapshot logging utilities."""

import pickle
from unittest.mock import patch

from tests.common import (
    pytest,
    create_sample_tensors,
    create_attack_config,
    create_nested_attack_config,
    verify_pickle_snapshot,
    verify_json_metadata,
)

from src.attack_utils.attack_snapshots import (
    save_attack_snapshot,
    load_attack_snapshot,
    list_attack_snapshots,
    get_snapshot_summary,
)


# =============================================================================
# TEST SUITE
# =============================================================================


class TestSaveAttackSnapshot:
    """Test suite for save_attack_snapshot function."""

    def test_save_snapshot_pickle_format(self, tmp_path):
        """Test saving snapshot in pickle format."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping", flip_fraction=0.7)

        save_attack_snapshot(
            client_id=0,
            round_num=3,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        snapshot_path = (
            tmp_path
            / "attack_snapshots_0"
            / "client_0"
            / "round_3"
            / "label_flipping.pickle"
        )
        verify_pickle_snapshot(
            snapshot_path,
            expected_client_id=0,
            expected_round=3,
            expected_attack_type="label_flipping",
            expected_num_samples=5,
        )

    def test_save_snapshot_respects_max_samples(self, tmp_path):
        """Test that max_samples parameter limits saved data."""
        data, labels = create_sample_tensors(batch_size=10)
        attack_config = create_attack_config("label_flipping")

        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            max_samples=3,
            save_format="pickle",
        )

        snapshot_path = (
            tmp_path
            / "attack_snapshots_0"
            / "client_0"
            / "round_1"
            / "label_flipping.pickle"
        )
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        # Should only save 3 samples, not 10
        assert len(snapshot["data"]) == 3
        assert len(snapshot["labels"]) == 3
        assert snapshot["metadata"]["num_samples"] == 3

    def test_save_snapshot_handles_nested_config(self, tmp_path):
        """Test saving snapshot with nested attack config (schedule-style)."""
        data, labels = create_sample_tensors(batch_size=5)
        # Nested config has "type" instead of "attack_type"
        attack_config = create_nested_attack_config("label_flipping", flip_fraction=0.5)

        save_attack_snapshot(
            client_id=2,
            round_num=7,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        snapshot_path = (
            tmp_path
            / "attack_snapshots_0"
            / "client_2"
            / "round_7"
            / "label_flipping.pickle"
        )
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        # Should extract "type" from nested config
        assert snapshot["metadata"]["attack_type"] == "label_flipping"
        assert snapshot["metadata"]["attack_config"] == attack_config

    def test_save_snapshot_creates_directory(self, tmp_path):
        """Test that save_attack_snapshot creates snapshots directory."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Directory should not exist initially
        snapshots_dir = tmp_path / "attack_snapshots_0"
        assert not snapshots_dir.exists()

        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Directory should be created
        assert snapshots_dir.exists()
        assert snapshots_dir.is_dir()

    def test_save_snapshot_overwrites_existing(self, tmp_path):
        """Test that saving snapshot overwrites existing file."""
        data1, labels1 = create_sample_tensors(batch_size=3)
        data2, labels2 = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Save first snapshot
        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data1,
            labels_sample=labels1,
            original_labels_sample=labels1.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Save second snapshot with same client/round (overwrite)
        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data2,
            labels_sample=labels2,
            original_labels_sample=labels2.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Should have latest data (5 samples, not 3)
        snapshot_path = (
            tmp_path
            / "attack_snapshots_0"
            / "client_0"
            / "round_1"
            / "label_flipping.pickle"
        )
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        assert snapshot["metadata"]["num_samples"] == 5

    @patch("src.attack_utils.attack_snapshots.pickle.dump")
    @patch("src.attack_utils.attack_snapshots.logging")
    def test_save_snapshot_handles_exception(
        self, mock_logging, mock_pickle_dump, tmp_path
    ):
        """Test that exceptions are caught and logged."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Make pickle.dump raise an exception
        mock_pickle_dump.side_effect = Exception("Simulated save error")

        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Should log warning about failure
        mock_logging.warning.assert_called()

    @pytest.mark.parametrize(
        "batch_size,max_samples,expected_samples",
        [
            (10, 5, 5),  # Batch larger than max
            (3, 5, 3),  # Batch smaller than max
            (5, 5, 5),  # Batch equals max
        ],
    )
    def test_save_snapshot_max_samples_variations(
        self, tmp_path, batch_size, max_samples, expected_samples
    ):
        """Test max_samples behavior with different batch sizes."""
        data, labels = create_sample_tensors(batch_size=batch_size)
        attack_config = create_attack_config("label_flipping")

        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            max_samples=max_samples,
            save_format="pickle",
        )

        snapshot_path = (
            tmp_path
            / "attack_snapshots_0"
            / "client_0"
            / "round_1"
            / "label_flipping.pickle"
        )
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        assert len(snapshot["data"]) == expected_samples

    def test_save_snapshot_preserves_attack_parameters(self, tmp_path):
        """Test that all attack parameters are preserved in snapshot."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config(
            "label_flipping",
            flip_fraction=0.7,
            target_class=5,
            source_class=3,
        )

        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        snapshot_path = (
            tmp_path
            / "attack_snapshots_0"
            / "client_0"
            / "round_1"
            / "label_flipping.pickle"
        )
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        saved_config = snapshot["metadata"]["attack_config"]
        assert saved_config["flip_fraction"] == 0.7
        assert saved_config["target_class"] == 5
        assert saved_config["source_class"] == 3


class TestLoadAttackSnapshot:
    """Test suite for load_attack_snapshot function."""

    def test_load_pickle_snapshot(self, tmp_path):
        """Test loading a pickle snapshot."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Save snapshot first
        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Load snapshot
        snapshot_path = (
            tmp_path
            / "attack_snapshots_0"
            / "client_0"
            / "round_1"
            / "label_flipping.pickle"
        )
        snapshot = load_attack_snapshot(str(snapshot_path))

        assert snapshot is not None
        assert "metadata" in snapshot
        assert "data" in snapshot
        assert "labels" in snapshot
        assert snapshot["metadata"]["client_id"] == 0
        assert snapshot["metadata"]["round_num"] == 1

    def test_load_nonexistent_snapshot(self):
        """Test loading a snapshot that doesn't exist."""
        snapshot = load_attack_snapshot("/nonexistent/path/snapshot.pickle")
        assert snapshot is None

    @patch("src.attack_utils.attack_snapshots.logging")
    def test_load_unsupported_format(self, mock_logging, tmp_path):
        """Test loading snapshot with unsupported format."""
        # Create file with unsupported extension
        invalid_path = tmp_path / "snapshot.txt"
        invalid_path.write_text("invalid format")

        snapshot = load_attack_snapshot(str(invalid_path))

        assert snapshot is None
        mock_logging.error.assert_called()

    @patch("src.attack_utils.attack_snapshots.logging")
    def test_load_corrupted_pickle(self, mock_logging, tmp_path):
        """Test loading corrupted pickle file."""
        # Create corrupted pickle file
        corrupted_path = tmp_path / "corrupted.pickle"
        corrupted_path.write_bytes(b"corrupted data")

        snapshot = load_attack_snapshot(str(corrupted_path))

        assert snapshot is None
        mock_logging.error.assert_called()

    @patch("src.attack_utils.attack_snapshots.logging")
    def test_load_corrupted_json(self, mock_logging, tmp_path):
        """Test loading corrupted JSON file."""
        # Create corrupted JSON file
        corrupted_path = tmp_path / "corrupted.json"
        corrupted_path.write_text("{invalid json")

        snapshot = load_attack_snapshot(str(corrupted_path))

        assert snapshot is None
        mock_logging.error.assert_called()


class TestListAttackSnapshots:
    """Test suite for list_attack_snapshots function."""

    def test_list_snapshots_empty_directory(self, tmp_path):
        """Test listing snapshots in empty directory."""
        snapshots = list_attack_snapshots(str(tmp_path))
        assert snapshots == []

    def test_list_snapshots_nonexistent_directory(self, tmp_path):
        """Test listing snapshots in nonexistent directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        snapshots = list_attack_snapshots(str(nonexistent_dir))
        assert snapshots == []

    def test_list_snapshots_multiple_files(self, tmp_path):
        """Test listing multiple snapshot files."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Create multiple snapshots
        for client_id in range(3):
            for round_num in range(2):
                save_attack_snapshot(
                    client_id=client_id,
                    round_num=round_num,
                    attack_config=attack_config,
                    data_sample=data,
                    labels_sample=labels,
                    original_labels_sample=labels.clone(),
                    output_dir=str(tmp_path),
                    save_format="pickle",
                )

        snapshots = list_attack_snapshots(str(tmp_path))

        # Should have 3 clients * 2 rounds = 6 snapshots
        assert len(snapshots) == 6

    def test_list_snapshots_mixed_formats(self, tmp_path):
        """Test listing snapshots with mixed pickle/JSON formats."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Create pickle snapshot
        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Create JSON snapshot
        save_attack_snapshot(
            client_id=1,
            round_num=2,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="json",
        )

        snapshots = list_attack_snapshots(str(tmp_path))

        # Only pickle files are listed (JSON files not included in list_attack_snapshots)
        assert len(snapshots) == 1

    def test_list_snapshots_ignores_other_files(self, tmp_path):
        """Test that list_attack_snapshots ignores non-snapshot files."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Create valid snapshot
        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Create non-snapshot files in snapshots directory
        snapshots_dir = tmp_path / "attack_snapshots_0"
        (snapshots_dir / "other_file.txt").write_text("not a snapshot")
        (snapshots_dir / "README.md").write_text("documentation")

        snapshots = list_attack_snapshots(str(tmp_path))

        # Should only list valid snapshot files
        assert len(snapshots) == 1

    def test_list_snapshots_sorted_order(self, tmp_path):
        """Test that snapshots are returned in sorted order."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Create snapshots in non-sequential order
        for client_id, round_num in [(2, 5), (0, 1), (1, 3)]:
            save_attack_snapshot(
                client_id=client_id,
                round_num=round_num,
                attack_config=attack_config,
                data_sample=data,
                labels_sample=labels,
                original_labels_sample=labels.clone(),
                output_dir=str(tmp_path),
                save_format="pickle",
            )

        snapshots = list_attack_snapshots(str(tmp_path))

        # Should be sorted
        filenames = [s.name for s in snapshots]
        assert filenames == sorted(filenames)


class TestGetSnapshotSummary:
    """Test suite for get_snapshot_summary function."""

    def test_summary_empty_directory(self, tmp_path):
        """Test summary for empty directory."""
        summary = get_snapshot_summary(str(tmp_path))

        assert summary["total_snapshots"] == 0
        assert summary["clients_attacked"] == []
        assert summary["rounds_with_attacks"] == []
        assert summary["attack_types"] == []

    def test_summary_single_snapshot(self, tmp_path):
        """Test summary with single snapshot."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        summary = get_snapshot_summary(str(tmp_path))

        assert summary["total_snapshots"] == 1
        assert summary["clients_attacked"] == [0]
        assert summary["rounds_with_attacks"] == [1]
        assert summary["attack_types"] == ["label_flipping"]

    def test_summary_multiple_clients_and_rounds(self, tmp_path):
        """Test summary with multiple clients and rounds."""
        data, labels = create_sample_tensors(batch_size=5)

        # Client 0: label_flipping in rounds 1, 2
        for round_num in [1, 2]:
            save_attack_snapshot(
                client_id=0,
                round_num=round_num,
                attack_config=create_attack_config("label_flipping"),
                data_sample=data,
                labels_sample=labels,
                original_labels_sample=labels.clone(),
                output_dir=str(tmp_path),
                save_format="pickle",
            )

        # Client 1: gaussian_noise in round 3
        save_attack_snapshot(
            client_id=1,
            round_num=3,
            attack_config=create_attack_config("gaussian_noise"),
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        summary = get_snapshot_summary(str(tmp_path))

        assert summary["total_snapshots"] == 3
        assert sorted(summary["clients_attacked"]) == [0, 1]
        assert sorted(summary["rounds_with_attacks"]) == [1, 2, 3]
        assert sorted(summary["attack_types"]) == ["gaussian_noise", "label_flipping"]

    def test_summary_deduplicates_attack_types(self, tmp_path):
        """Test that summary deduplicates attack types."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Multiple snapshots with same attack type
        for client_id in range(3):
            save_attack_snapshot(
                client_id=client_id,
                round_num=1,
                attack_config=attack_config,
                data_sample=data,
                labels_sample=labels,
                original_labels_sample=labels.clone(),
                output_dir=str(tmp_path),
                save_format="pickle",
            )

        summary = get_snapshot_summary(str(tmp_path))

        # Should only list attack type once
        assert summary["attack_types"] == ["label_flipping"]

    def test_summary_handles_nested_config(self, tmp_path):
        """Test summary handles nested attack config format."""
        data, labels = create_sample_tensors(batch_size=5)
        # Nested config with "type" instead of "attack_type"
        attack_config = create_nested_attack_config("label_flipping")

        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        summary = get_snapshot_summary(str(tmp_path))

        assert summary["attack_types"] == ["label_flipping"]

    def test_summary_handles_corrupted_snapshots(self, tmp_path):
        """Test summary handles corrupted snapshots gracefully."""
        data, labels = create_sample_tensors(batch_size=5)
        attack_config = create_attack_config("label_flipping")

        # Create valid snapshot
        save_attack_snapshot(
            client_id=0,
            round_num=1,
            attack_config=attack_config,
            data_sample=data,
            labels_sample=labels,
            original_labels_sample=labels.clone(),
            output_dir=str(tmp_path),
            save_format="pickle",
        )

        # Create corrupted snapshot in hierarchical structure
        snapshots_dir = tmp_path / "attack_snapshots_0"
        corrupted_dir = snapshots_dir / "client_1" / "round_2"
        corrupted_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = corrupted_dir / "label_flipping.pickle"
        corrupted_path.write_bytes(b"corrupted data")

        summary = get_snapshot_summary(str(tmp_path))

        # Should still count valid snapshot, skip corrupted
        assert summary["total_snapshots"] == 2  # Both files counted
        assert summary["clients_attacked"] == [0]  # Only valid one processed

    def test_summary_sorted_lists(self, tmp_path):
        """Test that summary returns sorted lists."""
        data, labels = create_sample_tensors(batch_size=5)

        # Create snapshots in non-sequential order
        for client_id, round_num in [(2, 5), (0, 1), (1, 3)]:
            save_attack_snapshot(
                client_id=client_id,
                round_num=round_num,
                attack_config=create_attack_config("label_flipping"),
                data_sample=data,
                labels_sample=labels,
                original_labels_sample=labels.clone(),
                output_dir=str(tmp_path),
                save_format="pickle",
            )

        summary = get_snapshot_summary(str(tmp_path))

        # All lists should be sorted
        assert summary["clients_attacked"] == sorted(summary["clients_attacked"])
        assert summary["rounds_with_attacks"] == sorted(summary["rounds_with_attacks"])
        assert summary["attack_types"] == sorted(summary["attack_types"])
