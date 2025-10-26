"""
Common test utilities and imports for FL framework testing.

Separation of concerns:
- tests/common.py: General utilities, imports, FL helpers (this file)
- tests/conftest.py: pytest-specific fixtures and configuration only
"""

# =============================================================================
# CONSOLIDATED IMPORTS
# =============================================================================

# Standard library
import sys
import os
import io
import locale
import logging
import contextlib
import inspect
from typing import Generator, List, Tuple, Any, Dict, Optional
from unittest.mock import Mock

# Third-party imports (conditional to avoid import errors)
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pytest
except ImportError:
    pytest = None

# Flower imports (conditional)
try:
    from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.client_proxy import ClientProxy
except ImportError:
    FitRes = None
    ndarrays_to_parameters = None
    parameters_to_ndarrays = None
    ClientProxy = None

# Type definitions
NDArray = Any  # Will be np.ndarray when numpy is available
Config = Dict[str, Any]
Metrics = Dict[str, Any]

# =============================================================================
# UNICODE UTILITIES (Cross-platform support)
# =============================================================================


def setup_unicode_output() -> None:
    """Configure stdout/stderr for Unicode output on Windows."""
    if sys.platform.startswith("win"):
        # Windows: Reconfigure stdout/stderr with UTF-8 encoding
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )

    # Set environment variable as fallback
    if "PYTHONIOENCODING" not in os.environ:
        os.environ["PYTHONIOENCODING"] = "utf-8"


@contextlib.contextmanager
def unicode_safe_output() -> Generator[None, None, None]:
    """Context manager for temporary Unicode-safe output."""
    if sys.platform.startswith("win") and hasattr(sys.stdout, "buffer"):
        # Store original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            # Temporarily reconfigure for Unicode
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )
            yield
        finally:
            # Restore original configuration
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    else:
        # Non-Windows platforms handle Unicode correctly by default
        yield


def safe_print(*args, **kwargs) -> None:
    """Print function that safely handles Unicode content via logging."""
    logger = logging.getLogger(__name__)
    message = " ".join(str(arg) for arg in args)
    logger.info(message)


def check_unicode_support() -> bool:
    """Check if current terminal supports Unicode output."""
    try:
        # Try to encode a simple emoji
        test_emoji = "🎭"
        if sys.platform.startswith("win"):
            # On Windows, check if we can encode to console's encoding
            encoding = locale.getpreferredencoding()
            test_emoji.encode(encoding)
        else:
            # On Unix-like systems, check UTF-8 support
            test_emoji.encode("utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def init_demo_output() -> None:
    """Initialize demo script output with Unicode support."""
    setup_unicode_output()
    logger = logging.getLogger(__name__)

    # Optional: Log environment info for debugging
    if os.environ.get("DEBUG_UNICODE"):
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Default encoding: {sys.getdefaultencoding()}")
        logger.info(f"Stdout encoding: {getattr(sys.stdout, 'encoding', 'unknown')}")
        logger.info(f"Unicode support: {check_unicode_support()}")
        logger.info("-" * 50)


def setup_test_logging(
    level: int = logging.INFO,
    format_string: str = "%(levelname)s - %(message)s",
    include_timestamp: bool = False,
) -> logging.Logger:
    """Configure logging for test scripts."""
    if include_timestamp:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, format=format_string)

    # Get logger for the calling module
    frame = inspect.currentframe()
    if frame and frame.f_back:
        module_name = frame.f_back.f_globals.get("__name__", __name__)
    else:
        module_name = __name__
    return logging.getLogger(module_name)


def init_test_environment(
    setup_logging: bool = True, include_timestamp: bool = False
) -> logging.Logger:
    """Initialize test environment with Unicode and logging setup."""
    # Set up Unicode support first
    setup_unicode_output()

    # Configure logging if requested
    if setup_logging:
        return setup_test_logging(include_timestamp=include_timestamp)
    else:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            module_name = frame.f_back.f_globals.get("__name__", __name__)
        else:
            module_name = __name__
        return logging.getLogger(module_name)


# =============================================================================
# FL TESTING UTILITIES
# =============================================================================


def generate_mock_client_data(
    num_clients: int, param_shape: Tuple[int, int] = (10, 5)
) -> "List[Tuple[Any, Any]]":
    """Generate mock client results (ClientProxy, FitRes)."""
    results: "List[Tuple[Any, Any]]" = []
    if np is None:
        raise ImportError("numpy is required for generate_mock_client_data")
    rng = np.random.default_rng(42)

    for i in range(num_clients):
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = str(i)

        # Create varied mock parameters
        if i < 2:  # Similar parameters for first two clients
            mock_params = [
                rng.standard_normal(param_shape) * 0.1,
                rng.standard_normal(param_shape[1]) * 0.1,
            ]
        else:  # Different parameters for remaining clients
            mock_params = [
                rng.standard_normal(param_shape) * (i + 1),
                rng.standard_normal(param_shape[1]) * (i + 1),
            ]

        fit_res = Mock(spec=FitRes)
        if ndarrays_to_parameters is None:
            raise ImportError("flwr is required for generate_mock_client_data")
        fit_res.parameters = ndarrays_to_parameters(mock_params)
        fit_res.num_examples = 100
        fit_res.metrics = {"accuracy": 0.8 + i * 0.01, "loss": 0.5 - i * 0.02}

        results.append((client_proxy, fit_res))

    return results


class FLTestHelpers:
    """Collection of FL testing helper methods and utilities."""

    @staticmethod
    def create_mock_flower_client(client_id: int) -> Mock:
        """Create a mock Flower client for testing."""
        client = Mock()
        client.cid = str(client_id)
        return client

    @staticmethod
    def assert_valid_fl_result(
        result: Any, expected_shape: "Optional[Tuple[int, ...]]" = None
    ) -> None:
        """Validate FL aggregation result structure and content."""
        assert result is not None, "FL result should not be None"

        if expected_shape and hasattr(result, "shape"):
            assert result.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {result.shape}"
            )

    @staticmethod
    def create_byzantine_clients(num_clients: int, byzantine_count: int) -> List[int]:
        """Generate indices for Byzantine (malicious) clients."""
        if byzantine_count > num_clients:
            raise ValueError("Byzantine count cannot exceed total clients")

        # Last N clients are Byzantine
        return list(range(num_clients - byzantine_count, num_clients))

    @staticmethod
    def validate_aggregation_invariants(
        client_results: "List[Tuple[Any, Any]]", aggregated_result: Any
    ) -> None:
        """Validate common FL aggregation invariants."""
        assert len(client_results) > 0, "Should have client results to aggregate"
        assert aggregated_result is not None, "Aggregated result should not be None"


def assert_valid_fl_result(
    result: Any, expected_shape: "Optional[Tuple[int, ...]]" = None
) -> None:
    """Convenience function for FL result validation."""
    FLTestHelpers.assert_valid_fl_result(result, expected_shape)


def create_mock_flower_client(client_id: int) -> Mock:
    """Convenience function for creating mock Flower clients."""
    return FLTestHelpers.create_mock_flower_client(client_id)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Common test configuration values
DEFAULT_PARAM_SHAPE = (10, 5)
DEFAULT_NUM_CLIENTS = 5
DEFAULT_NUM_EXAMPLES = 100

# Strategy configurations for testing
STRATEGY_CONFIGS = {
    "trust": {
        "aggregation_strategy_keyword": "trust",
        "num_of_rounds": 5,
        "num_of_clients": 10,
        "trust_threshold": 0.7,
        "beta_value": 0.5,
        "config_is_ai_generated": False,
    },
    "pid": {
        "aggregation_strategy_keyword": "pid",
        "num_of_rounds": 3,
        "num_of_clients": 8,
        "Kp": 1.0,
        "Ki": 0.1,
        "Kd": 0.01,
        "config_is_ai_generated": False,
    },
    "krum": {
        "aggregation_strategy_keyword": "krum",
        "num_of_rounds": 4,
        "num_of_clients": 12,
        "num_krum_selections": 8,
        "config_is_ai_generated": False,
    },
    "multi-krum": {
        "aggregation_strategy_keyword": "multi-krum",
        "num_of_rounds": 4,
        "num_of_clients": 12,
        "num_krum_selections": 8,
        "config_is_ai_generated": False,
    },
    "trimmed_mean": {
        "aggregation_strategy_keyword": "trimmed_mean",
        "num_of_rounds": 4,
        "num_of_clients": 10,
        "trim_ratio": 0.2,
        "config_is_ai_generated": False,
    },
}

# =============================================================================
# RE-EXPORTS FOR CONVENIENCE
# =============================================================================

# Re-export common imports for easy access
__all__ = [
    # Standard imports
    "Mock",
    "np",
    "pytest",
    # Flower imports
    "FitRes",
    "ndarrays_to_parameters",
    "parameters_to_ndarrays",
    "ClientProxy",
    # Type definitions
    "NDArray",
    "Config",
    "Metrics",
    # Unicode utilities
    "setup_unicode_output",
    "unicode_safe_output",
    "safe_print",
    "check_unicode_support",
    "init_demo_output",
    "setup_test_logging",
    "init_test_environment",
    # FL testing utilities
    "generate_mock_client_data",
    "FLTestHelpers",
    "assert_valid_fl_result",
    "create_mock_flower_client",
    # Constants
    "DEFAULT_PARAM_SHAPE",
    "DEFAULT_NUM_CLIENTS",
    "DEFAULT_NUM_EXAMPLES",
    "STRATEGY_CONFIGS",
]
