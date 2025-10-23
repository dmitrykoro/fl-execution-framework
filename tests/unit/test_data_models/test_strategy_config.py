import json
from typing import Any, Dict

from src.data_models.simulation_strategy_config import StrategyConfig


class TestStrategyConfig:
    """Test StrategyConfig data model."""

    def test_init_with_valid_parameters(self) -> None:
        """Test initialization with valid parameters."""
        config = StrategyConfig(
            aggregation_strategy_keyword="trust",
            num_of_rounds=5,
            num_of_clients=10,
            trust_threshold=0.7,
            beta_value=0.5,
        )

        assert config.aggregation_strategy_keyword == "trust"
        assert config.num_of_rounds == 5
        assert config.num_of_clients == 10
        assert config.trust_threshold == 0.7
        assert config.beta_value == 0.5

    def test_init_with_string_booleans(self):
        """Test initialization converts string booleans correctly."""
        config = StrategyConfig(
            remove_clients="true",
            show_plots="false",
            save_plots="true",
            save_csv="false",
        )

        assert config.remove_clients is True
        assert config.show_plots is False
        assert config.save_plots is True
        assert config.save_csv is False

    def test_init_with_none_values(self):
        """Test initialization with None values."""
        config = StrategyConfig()

        assert config.aggregation_strategy_keyword is None
        assert config.num_of_rounds is None
        assert config.num_of_clients is None
        assert config.trust_threshold is None

    def test_init_with_mixed_parameters(self):
        """Test initialization with mixed parameter types."""
        config = StrategyConfig(
            aggregation_strategy_keyword="pid",
            num_of_rounds=3,
            remove_clients="true",
            Kp=1.0,
            Ki=0.1,
            Kd=0.01,
            show_plots="false",
        )

        assert config.aggregation_strategy_keyword == "pid"
        assert config.num_of_rounds == 3
        assert config.remove_clients is True
        assert config.Kp == 1.0
        assert config.Ki == 0.1
        assert config.Kd == 0.01
        assert config.show_plots is False

    def test_from_dict_valid_config(self) -> None:
        """Test from_dict with valid configuration dictionary."""
        config_dict: Dict[str, Any] = {
            "aggregation_strategy_keyword": "krum",
            "num_of_rounds": 4,
            "num_of_clients": 8,
            "num_krum_selections": 3,
            "remove_clients": "true",
        }

        config = StrategyConfig.from_dict(config_dict)

        assert config.aggregation_strategy_keyword == "krum"
        assert config.num_of_rounds == 4
        assert config.num_of_clients == 8
        assert config.num_krum_selections == 3
        assert config.remove_clients is True

    def test_from_dict_empty_config(self) -> None:
        """Test from_dict with empty dictionary."""
        config = StrategyConfig.from_dict({})

        assert config.aggregation_strategy_keyword is None
        assert config.num_of_rounds is None
        assert config.num_of_clients is None

    def test_from_dict_with_string_booleans(self):
        """Test from_dict converts string booleans correctly."""
        config_dict = {
            "save_plots": "true",
            "preserve_dataset": "false",
            "show_plots": "true",
        }

        config = StrategyConfig.from_dict(config_dict)

        assert config.save_plots is True
        assert config.preserve_dataset is False
        assert config.show_plots is True

    def test_to_json_valid_config(self) -> None:
        """Test to_json serialization."""
        config = StrategyConfig(
            aggregation_strategy_keyword="trust",
            num_of_rounds=5,
            num_of_clients=10,
            trust_threshold=0.7,
            remove_clients=True,
        )

        json_str = config.to_json()
        parsed_json = json.loads(json_str)

        assert parsed_json["aggregation_strategy_keyword"] == "trust"
        assert parsed_json["num_of_rounds"] == 5
        assert parsed_json["num_of_clients"] == 10
        assert parsed_json["trust_threshold"] == 0.7
        assert parsed_json["remove_clients"] is True

    def test_to_json_with_none_values(self):
        """Test to_json with None values."""
        config = StrategyConfig(
            aggregation_strategy_keyword="pid", num_of_rounds=None, Kp=1.5
        )

        json_str = config.to_json()
        parsed_json = json.loads(json_str)

        assert parsed_json["aggregation_strategy_keyword"] == "pid"
        assert parsed_json["num_of_rounds"] is None
        assert parsed_json["Kp"] == 1.5

    def test_to_json_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original_config = StrategyConfig(
            aggregation_strategy_keyword="multi-krum",
            num_of_rounds=6,
            num_of_clients=12,
            num_krum_selections=4,
            remove_clients=True,
            beta_value=0.8,
        )

        # Serialize to JSON
        json_str = original_config.to_json()

        # Deserialize back to dict and create new config
        config_dict = json.loads(json_str)
        new_config = StrategyConfig.from_dict(config_dict)

        # Compare all relevant fields
        assert (
            new_config.aggregation_strategy_keyword
            == original_config.aggregation_strategy_keyword
        )
        assert new_config.num_of_rounds == original_config.num_of_rounds
        assert new_config.num_of_clients == original_config.num_of_clients
        assert new_config.num_krum_selections == original_config.num_krum_selections
        assert new_config.remove_clients == original_config.remove_clients
        assert new_config.beta_value == original_config.beta_value

    def test_invalid_parameter_handling(self):
        """Test handling invalid parameters gracefully."""
        # StrategyConfig accepts any keyword arguments, so this tests that
        # unknown parameters are set as attributes
        config = StrategyConfig(unknown_parameter="test_value", another_unknown=123)

        assert config.unknown_parameter == "test_value"
        assert config.another_unknown == 123

    def test_boolean_conversion_edge_cases(self):
        """Test boolean conversion edge cases."""
        config = StrategyConfig(
            # Only "true" and "false" strings are converted
            actual_boolean=True,
            string_true="true",
            string_false="false",
            other_string="maybe",
            number_value=1,
        )

        assert config.actual_boolean is True
        assert config.string_true is True
        assert config.string_false is False
        assert config.other_string == "maybe"  # Not converted
        assert config.number_value == 1  # Not converted

    def test_strategy_specific_parameters(self):
        """Test strategy-specific parameters."""
        # Trust strategy parameters
        trust_config = StrategyConfig(
            aggregation_strategy_keyword="trust",
            trust_threshold=0.6,
            reputation_threshold=0.5,
            beta_value=0.3,
            num_of_clusters=3,
        )

        assert trust_config.trust_threshold == 0.6
        assert trust_config.reputation_threshold == 0.5
        assert trust_config.beta_value == 0.3
        assert trust_config.num_of_clusters == 3

        # PID strategy parameters
        pid_config = StrategyConfig(
            aggregation_strategy_keyword="pid", Kp=2.0, Ki=0.5, Kd=0.1, num_std_dev=2.5
        )

        assert pid_config.Kp == 2.0
        assert pid_config.Ki == 0.5
        assert pid_config.Kd == 0.1
        assert pid_config.num_std_dev == 2.5

        # Krum strategy parameters
        krum_config = StrategyConfig(
            aggregation_strategy_keyword="krum", num_krum_selections=5
        )

        assert krum_config.num_krum_selections == 5

        # Trimmed mean strategy parameters
        trimmed_config = StrategyConfig(
            aggregation_strategy_keyword="trimmed_mean", trim_ratio=0.2
        )

        assert trimmed_config.trim_ratio == 0.2

    def test_training_parameters(self):
        """Test training-related parameters."""
        config = StrategyConfig(
            training_subset_fraction=0.8,
            min_fit_clients=5,
            min_evaluate_clients=3,
            min_available_clients=8,
            num_of_client_epochs=2,
            batch_size=32,
            training_device="cuda",
            cpus_per_client=2,
            gpus_per_client=0.5,
        )

        assert config.training_subset_fraction == 0.8
        assert config.min_fit_clients == 5
        assert config.min_evaluate_clients == 3
        assert config.min_available_clients == 8
        assert config.num_of_client_epochs == 2
        assert config.batch_size == 32
        assert config.training_device == "cuda"
        assert config.cpus_per_client == 2
        assert config.gpus_per_client == 0.5

    def test_attack_parameters(self):
        """Test attack-related parameters."""
        config = StrategyConfig(
            num_of_malicious_clients=2,
            attack_type="gaussian_noise",
            attack_ratio=0.3,
            target_noise_snr=10.0,
        )

        assert config.num_of_malicious_clients == 2
        assert config.attack_type == "gaussian_noise"
        assert config.attack_ratio == 0.3
        assert config.target_noise_snr == 10.0
