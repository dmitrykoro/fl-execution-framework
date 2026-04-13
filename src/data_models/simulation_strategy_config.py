import json

from dataclasses import dataclass, asdict
from typing import Optional, Any


@dataclass
class StrategyConfig:

    aggregation_strategy_keyword: str = None
    remove_clients: bool = None
    begin_removing_from_round: int = None
    dataset_keyword: str = None
    num_of_rounds: int = None
    num_of_clients: int = None
    num_of_malicious_clients: int = None
    show_plots: bool = None
    save_plots: bool = None
    save_csv: bool = None
    save_attack_snapshots: bool = None
    attack_snapshot_format: str = "pickle"
    snapshot_max_samples: int = 5
    training_device: str = None
    cpus_per_client: float = None
    gpus_per_client: float = None

    trust_threshold: float = None
    reputation_threshold: float = None
    beta_value: float = None
    num_of_clusters: int = None

    Kp: float = None
    Ki: float = None
    Kd: float = None
    num_std_dev: float = None

    num_bulyan_benign_clients: int = None

    training_subset_fraction: float = None
    min_fit_clients: int = None
    min_evaluate_clients: int = None
    min_available_clients: int = None
    evaluate_metrics_aggregation_fn: str = None
    num_of_client_epochs: int = None
    batch_size: int = None
    preserve_dataset: bool = None

    num_krum_selections: int = None

    trim_ratio: float = None

    # PID-MADE parameters
    lam: float = None            # forgetting factor λ ∈ (0, 1)
    threshold_mode: str = None   # 'fixed' | 'chebyshev' | 'gaussian'
    tau: float = None            # fixed threshold (threshold_mode == 'fixed')
    alpha: float = None          # false-positive rate (threshold_mode in chebyshev/gaussian)

    # FreqFed parameters
    min_cluster_size: int = None  # HDBSCAN min_cluster_size (default 5)

    strict_mode: bool = None

    strategy_number: int = None

    attack_schedule: list = None

    config_is_ai_generated: bool = False
    user_prompt_for_ai_generated: str = None
    ai_agent_config_reasoning: str = None
    ai_configuration_error: str = None
    num_ai_config_corrections: int = None

    attack_type: str = None
    attack_ratio: float = None
    target_noise_snr: float = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if value in ("true", "false"):
                setattr(self, key, value == "true")
            elif key == "training_device" and value == "gpu":
                setattr(self, key, "cuda")
            else:
                setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Allow access to dynamically set attributes"""
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def from_dict(cls, strategy_config: dict):
        """Create config instance from dict"""
        return cls(**strategy_config)

    def to_json(self):
        """Convert config to json"""
        return json.dumps(asdict(self))
