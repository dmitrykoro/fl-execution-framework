"""
Client configuration for federated learning strategies.

This module automatically configures client participation parameters based on
the aggregation strategy keywords to ensure proper behavior.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClientConfigRecommendation:
    """Recommendation for client configuration parameters."""
    min_fit_clients: int
    min_evaluate_clients: int
    min_available_clients: int
    reasoning: str
    warning_message: Optional[str] = None


# Strategies that REQUIRE consistent client participation
CONSISTENT_PARTICIPATION_STRATEGIES = {
    "trust": "Trust-based strategies need consistent client participation to build reliable reputation scores",
    "pid": "PID controllers require consistent feedback loops for stable convergence",
    "pid_scaled": "Scaled PID controllers need consistent client participation for proper scaling",
    "pid_standardized": "Standardized PID controllers require consistent participation for normalization",
    "krum": "Krum requires all clients for accurate distance calculations and Byzantine detection",
    "multi-krum": "Multi-Krum needs consistent client pool for reliable Byzantine-robust aggregation",
    "multi-krum-based": "Multi-Krum based strategies require consistent participation for effectiveness",
    "rfa": "RFA (Robust Federated Averaging) needs all clients for proper Byzantine fault tolerance",
    "bulyan": "Bulyan requires specific client ratios and consistent participation for Byzantine robustness"
}

# Strategies that work with variable participation
FLEXIBLE_PARTICIPATION_STRATEGIES = {
    "trimmed_mean": "Trimmed Mean can work with variable participation but benefits from consistency"
}


def analyze_client_requirements(
    strategy_keyword: str,
    num_of_clients: int,
    num_of_malicious_clients: int = 0
) -> ClientConfigRecommendation:
    """
    Analyze strategy requirements and recommend client configuration.

    Args:
        strategy_keyword: The aggregation strategy being used
        num_of_clients: Total number of clients available
        num_of_malicious_clients: Number of malicious clients in experiment

    Returns:
        ClientConfigRecommendation with suggested parameters and reasoning
    """

    if strategy_keyword in CONSISTENT_PARTICIPATION_STRATEGIES:
        # These strategies REQUIRE all clients for proper operation
        reasoning = CONSISTENT_PARTICIPATION_STRATEGIES[strategy_keyword]
        return ClientConfigRecommendation(
            min_fit_clients=num_of_clients,
            min_evaluate_clients=num_of_clients,
            min_available_clients=num_of_clients,
            reasoning=f"AUTO-CONFIGURED: {reasoning}",
            warning_message=f"INFO: {strategy_keyword} requires all clients to participate for proper operation. "
                           f"Set min_clients = {num_of_clients} for optimal results."
        )

    elif strategy_keyword in FLEXIBLE_PARTICIPATION_STRATEGIES:
        # These can work with variable participation but warn about benefits of consistency
        min_clients = max(3, int(num_of_clients * 0.8))  # Use 80% as default
        return ClientConfigRecommendation(
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=num_of_clients,
            reasoning=f"{FLEXIBLE_PARTICIPATION_STRATEGIES[strategy_keyword]}",
            warning_message=f"TIP: {strategy_keyword} works with variable participation but may perform better "
                           f"with consistent participation (min_clients = {num_of_clients})"
        )

    else:
        # Unknown strategy - use conservative defaults
        min_clients = max(3, int(num_of_clients * 0.6))  # Conservative 60%
        return ClientConfigRecommendation(
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=num_of_clients,
            reasoning="Conservative defaults for unknown strategy",
            warning_message=f"Unknown strategy '{strategy_keyword}'. Using conservative defaults. "
                           f"Consider researching whether this strategy needs consistent client participation."
        )


def apply_client_config(config_dict: Dict, verbose: bool = True) -> Dict:
    """
    Apply client configuration to a strategy config dictionary.

    Args:
        config_dict: Dictionary containing strategy configuration
        verbose: Whether to print recommendation messages

    Returns:
        Updated config dictionary with client parameters
    """

    strategy_keyword = config_dict.get("aggregation_strategy_keyword")
    num_of_clients = config_dict.get("num_of_clients")
    num_malicious = config_dict.get("num_of_malicious_clients", 0)

    if not strategy_keyword or not num_of_clients:
        logger.warning("Cannot apply client config: missing strategy_keyword or num_of_clients")
        return config_dict

    # Check if user explicitly configured client parameters
    user_configured = any(config_dict.get(param) is not None for param in
                         ["min_fit_clients", "min_evaluate_clients", "min_available_clients"])

    # Get recommendation
    recommendation = analyze_client_requirements(
        strategy_keyword=strategy_keyword,
        num_of_clients=num_of_clients,
        num_of_malicious_clients=num_malicious
    )

    # Apply recommendations if not user-configured
    if not user_configured:
        config_dict["min_fit_clients"] = recommendation.min_fit_clients
        config_dict["min_evaluate_clients"] = recommendation.min_evaluate_clients
        config_dict["min_available_clients"] = recommendation.min_available_clients

        if verbose:
            print(f"\nAUTO-CONFIGURED:")
            print(f"   Strategy: {strategy_keyword}")
            print(f"   Applied: min_fit={recommendation.min_fit_clients}, min_eval={recommendation.min_evaluate_clients}, total={num_of_clients}")
            print(f"   REASON: {recommendation.reasoning.replace('AUTO-CONFIGURED: ', '')}")

    else:
        # User configured manually - check if config matches recommendations
        if verbose:
            current_fit = config_dict.get("min_fit_clients", "not set")
            current_eval = config_dict.get("min_evaluate_clients", "not set")

            # Check if manual config matches optimal settings
            is_optimal = (current_fit == recommendation.min_fit_clients and
                         current_eval == recommendation.min_evaluate_clients)

            if is_optimal:
                print(f"\nMANUAL CONFIG VERIFIED:")
                print(f"   Strategy: {strategy_keyword}")
                print(f"   Current: min_fit={current_fit}, min_eval={current_eval}, total={num_of_clients}")
                print(f"   STATUS: Configuration is optimal for {strategy_keyword} strategy!")
            else:
                print(f"\nMANUAL CONFIG ISSUE:")
                print(f"   Strategy: {strategy_keyword}")
                print(f"   Current: min_fit={current_fit}, min_eval={current_eval}, total={num_of_clients}")
                print(f"   RECOMMENDATION: {recommendation.warning_message}")

    return config_dict


def validate_client_config(config_dict: Dict) -> List[str]:
    """
    Validate client configuration and return list of issues/warnings.

    Args:
        config_dict: Strategy configuration dictionary

    Returns:
        List of validation messages (empty if no issues)
    """
    issues = []

    strategy_keyword = config_dict.get("aggregation_strategy_keyword")
    num_of_clients = config_dict.get("num_of_clients", 0)
    min_fit = config_dict.get("min_fit_clients", 0)
    min_eval = config_dict.get("min_evaluate_clients", 0)
    min_available = config_dict.get("min_available_clients", 0)

    # Check for potential convergence issues
    if strategy_keyword in CONSISTENT_PARTICIPATION_STRATEGIES:
        if min_fit < num_of_clients or min_eval < num_of_clients:
            issues.append(
                f"WARNING - CONVERGENCE RISK: {strategy_keyword} may not converge properly with "
                f"min_fit_clients={min_fit} or min_evaluate_clients={min_eval} < num_of_clients={num_of_clients}. "
                f"Consider setting both to {num_of_clients} for reliable results."
            )

    # Check for statistical significance issues
    if min_fit < 3:
        issues.append(
            f"STATISTICAL WARNING: min_fit_clients={min_fit} may not provide statistically significant results. "
            f"Consider using at least 3 clients."
        )

    # Check for resource allocation issues
    if min_available > num_of_clients:
        issues.append(
            f"CONFIG ERROR: min_available_clients={min_available} > num_of_clients={num_of_clients}. "
            f"Cannot require more clients than available."
        )

    return issues