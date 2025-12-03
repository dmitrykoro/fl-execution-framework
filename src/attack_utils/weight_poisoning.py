"""
Weight-level poisoning attacks for FL model updates.

These attacks modify model parameters after local training but before
submission to the aggregation server. This differs from data poisoning
which corrupts training data before model training.

Supported attack types:
- model_poisoning: Targeted manipulation of a subset of weights
- gradient_scaling: Scale all weights by a factor (simulates gradient attack)
- byzantine_perturbation: Large random perturbations to weights
"""

import logging
from typing import Dict, List, Optional, Callable

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

# Attack types that operate on model weights (not training data)
WEIGHT_ATTACK_TYPES = frozenset(
    [
        "model_poisoning",
        "gradient_scaling",
        "byzantine_perturbation",
    ]
)


def apply_model_poisoning(
    parameters: List[NDArray],
    poison_ratio: float = 0.1,
    magnitude: float = 50.0,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """
    Apply targeted weight manipulation by spiking a subset of parameters.

    This attack selects a random subset of weights and multiplies them
    by a large magnitude, creating anomalous updates that can degrade
    model performance or inject backdoor behavior.

    Args:
        parameters: List of model parameter arrays (weights/biases).
        poison_ratio: Fraction of weights to poison (0.0 to 1.0).
        magnitude: Multiplier for poisoned weights.
        seed: Random seed for reproducibility.

    Returns:
        List of poisoned parameter arrays.

    References:
        - PoisonedFL (CVPR 2025): Multi-round consistency attacks
    """
    rng = np.random.default_rng(seed)
    poisoned_params = []

    for param in parameters:
        poisoned = param.copy()
        flat = poisoned.flatten()
        num_poison = max(1, int(len(flat) * poison_ratio))
        poison_indices = rng.choice(len(flat), size=num_poison, replace=False)
        flat[poison_indices] *= magnitude
        poisoned_params.append(flat.reshape(param.shape))

    logger.debug(
        f"Model poisoning applied: ratio={poison_ratio}, magnitude={magnitude}"
    )
    return poisoned_params


def apply_gradient_scaling(
    parameters: List[NDArray],
    scale_factor: float = 3.0,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """
    Scale all model parameters by a constant factor.

    This attack simulates the effect of gradient-based attacks by
    amplifying the magnitude of all weight updates, potentially
    causing the global model to diverge.

    Args:
        parameters: List of model parameter arrays.
        scale_factor: Multiplier for all weights.
        seed: Random seed (unused, kept for API consistency).

    Returns:
        List of scaled parameter arrays.

    References:
        - GradAttack (Princeton): Gradient inversion evaluation library
    """
    scaled_params = [param * scale_factor for param in parameters]

    logger.debug(f"Gradient scaling applied: scale_factor={scale_factor}")
    return scaled_params


def apply_byzantine_perturbation(
    parameters: List[NDArray],
    noise_scale: float = 15.0,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """
    Apply large random perturbations to model weights.

    This attack simulates Byzantine client behavior by replacing
    model weights with random values scaled to a large magnitude,
    representing adversarial or faulty client updates.

    Args:
        parameters: List of model parameter arrays.
        noise_scale: Standard deviation multiplier for random noise.
        seed: Random seed for reproducibility.

    Returns:
        List of perturbed parameter arrays.
    """
    rng = np.random.default_rng(seed)
    perturbed_params = []

    for param in parameters:
        noise = rng.standard_normal(param.shape) * noise_scale
        perturbed_params.append(noise.astype(param.dtype))

    logger.debug(f"Byzantine perturbation applied: noise_scale={noise_scale}")
    return perturbed_params


# Registry mapping attack type names to functions
_WEIGHT_ATTACK_FUNCTIONS: Dict[str, Callable] = {
    "model_poisoning": apply_model_poisoning,
    "gradient_scaling": apply_gradient_scaling,
    "byzantine_perturbation": apply_byzantine_perturbation,
}


def apply_weight_poisoning(
    parameters: List[NDArray],
    attack_configs: List[dict],
) -> List[NDArray]:
    """
    Apply weight-level poisoning attacks based on configuration.

    This is the main entry point for weight poisoning, dispatching
    to specific attack implementations based on attack_type.

    Args:
        parameters: List of model parameter arrays from local training.
        attack_configs: List of attack configuration dicts, each containing:
            - attack_type: One of WEIGHT_ATTACK_TYPES
            - Additional parameters specific to the attack type

    Returns:
        List of poisoned parameter arrays.

    Raises:
        ValueError: If attack_type is not a valid weight attack.
    """
    result = parameters

    for config in attack_configs:
        attack_type = config.get("attack_type")

        if attack_type not in WEIGHT_ATTACK_TYPES:
            continue  # Skip data poisoning attacks (handled elsewhere)

        attack_fn = _WEIGHT_ATTACK_FUNCTIONS.get(attack_type)
        if attack_fn is None:
            raise ValueError(f"Unknown weight attack type: {attack_type}")

        # Extract attack-specific parameters
        if attack_type == "model_poisoning":
            result = attack_fn(
                result,
                poison_ratio=config.get("poison_ratio", 0.1),
                magnitude=config.get("magnitude", 50.0),
                seed=config.get("seed"),
            )
        elif attack_type == "gradient_scaling":
            result = attack_fn(
                result,
                scale_factor=config.get("scale_factor", 3.0),
                seed=config.get("seed"),
            )
        elif attack_type == "byzantine_perturbation":
            result = attack_fn(
                result,
                noise_scale=config.get("noise_scale", 15.0),
                seed=config.get("seed"),
            )

    return result


def is_weight_attack(attack_type: str) -> bool:
    """Check if an attack type is a weight-level attack."""
    return attack_type in WEIGHT_ATTACK_TYPES
