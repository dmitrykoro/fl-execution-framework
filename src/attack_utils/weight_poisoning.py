"""Weight-level poisoning attacks for FL model updates."""

import logging
from typing import Dict, List, Optional, Callable

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

# Default threshold for overflow warnings
_MAX_SAFE_WEIGHT_VALUE = 1e6


def _check_and_warn_overflow(
    params: List[NDArray],
    attack_type: str,
    max_safe_value: float = _MAX_SAFE_WEIGHT_VALUE,
) -> List[NDArray]:
    """
    Detects and warns about numerical overflow in poisoned weights.

    Args:
        params: List of parameter arrays to check.
        attack_type: Name of the attack for logging context.
        max_safe_value: Threshold above which to warn.

    Returns:
        The input params unchanged.
    """
    for i, param in enumerate(params):
        max_val = np.max(np.abs(param))
        if max_val > max_safe_value:
            logger.warning(
                f"{attack_type}: param[{i}] has extreme value {max_val:.2e} "
                f"(exceeds {max_safe_value:.0e}). May cause loss explosion."
            )
        if not np.isfinite(param).all():
            logger.error(f"{attack_type}: param[{i}] contains NaN/Inf values!")
    return params


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
    magnitude: float = 5.0,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """
    Applies targeted weight manipulation to a subset of parameters.

    Args:
        parameters: List of model parameter arrays.
        poison_ratio: Fraction of weights to poison.
        magnitude: Number of standard deviations for poisoned values.
        seed: Random seed.

    Returns:
        List of poisoned parameter arrays.
    """
    rng = np.random.default_rng(seed)
    poisoned_params = []

    for param in parameters:
        poisoned = param.copy()
        flat = poisoned.flatten()
        num_poison = max(1, int(len(flat) * poison_ratio))
        poison_indices = rng.choice(len(flat), size=num_poison, replace=False)

        param_std = np.std(flat) + 1e-8
        poison_value = magnitude * param_std
        flat[poison_indices] = np.sign(flat[poison_indices]) * poison_value

        poisoned_params.append(flat.reshape(param.shape))

    logger.debug(
        f"Model poisoning applied: ratio={poison_ratio}, magnitude={magnitude} std"
    )
    return poisoned_params


def apply_gradient_scaling(
    parameters: List[NDArray],
    scale_factor: float = 2.0,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """
    Scales all model parameters by a constant factor.

    Args:
        parameters: List of model parameter arrays.
        scale_factor: Multiplier for all weights.
        seed: Random seed.

    Returns:
        List of scaled parameter arrays.
    """
    scaled_params = [param * scale_factor for param in parameters]

    logger.debug(f"Gradient scaling applied: scale_factor={scale_factor}")
    return scaled_params


def apply_byzantine_perturbation(
    parameters: List[NDArray],
    noise_scale: float = 3.0,
    seed: Optional[int] = None,
) -> List[NDArray]:
    """
    Applies random perturbations to model weights.

    Args:
        parameters: List of model parameter arrays.
        noise_scale: Noise magnitude as multiple of parameter std deviation.
        seed: Random seed.

    Returns:
        List of perturbed parameter arrays.
    """
    rng = np.random.default_rng(seed)
    perturbed_params = []

    for param in parameters:
        param_std = np.std(param) + 1e-8
        scaled_noise = rng.standard_normal(param.shape) * noise_scale * param_std
        perturbed = param + scaled_noise
        perturbed_params.append(perturbed.astype(param.dtype))

    logger.debug(f"Byzantine perturbation applied: noise_scale={noise_scale} std")
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
    Applies weight-level poisoning attacks based on configuration.

    Args:
        parameters: List of model parameter arrays.
        attack_configs: List of attack configuration dicts.

    Returns:
        List of poisoned parameter arrays.

    Raises:
        ValueError: If attack_type is not a valid weight attack.
    """
    result = parameters

    for config in attack_configs:
        attack_type = config.get("attack_type")

        if attack_type not in WEIGHT_ATTACK_TYPES:
            continue

        attack_fn = _WEIGHT_ATTACK_FUNCTIONS.get(attack_type)
        if attack_fn is None:
            raise ValueError(f"Unknown weight attack type: {attack_type}")

        if attack_type == "model_poisoning":
            result = attack_fn(
                result,
                poison_ratio=config.get("poison_ratio", 0.1),
                magnitude=config.get("magnitude", 5.0),
                seed=config.get("seed"),
            )
        elif attack_type == "gradient_scaling":
            result = attack_fn(
                result,
                scale_factor=config.get("scale_factor", 2.0),
                seed=config.get("seed"),
            )
        elif attack_type == "byzantine_perturbation":
            result = attack_fn(
                result,
                noise_scale=config.get("noise_scale", 3.0),
                seed=config.get("seed"),
            )

        _check_and_warn_overflow(result, attack_type)

    return result


def is_weight_attack(attack_type: str) -> bool:
    """Checks if an attack type is a weight-level attack."""
    return attack_type in WEIGHT_ATTACK_TYPES
