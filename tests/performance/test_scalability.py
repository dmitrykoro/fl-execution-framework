"""
Scalability tests for federated learning simulation framework.

Tests execution time measurement, computational complexity bounds,
and performance scaling across different configurations.
"""

import statistics
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from src.data_models.client_info import ClientInfo
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory

from tests.fixtures.mock_datasets import (
    MockFederatedDataset,
    generate_byzantine_client_parameters,
    generate_mock_client_parameters,
)


class PerformanceTimer:
    """Measure execution time and performance metrics."""

    def __init__(self) -> None:
        self.measurements: List[Tuple[str, float]] = []
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start timing measurement."""
        self.start_time = time.perf_counter()

    def stop(self, label: str = "") -> float:
        """Stop timing and record measurement."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        elapsed = time.perf_counter() - self.start_time
        self.measurements.append((label, elapsed))
        self.start_time = None
        return elapsed

    def get_measurements(self) -> List[Tuple[str, float]]:
        """Return all recorded measurements."""
        return self.measurements.copy()

    def get_average_time(self) -> float:
        """Return average execution time across all measurements."""
        if not self.measurements:
            return 0.0
        return statistics.mean([time for _, time in self.measurements])

    def get_total_time(self) -> float:
        """Return total execution time across all measurements."""
        return sum(time for _, time in self.measurements)


@pytest.fixture
def performance_timer() -> PerformanceTimer:
    """Provide performance timing capabilities."""
    return PerformanceTimer()


class TestClientScalability:
    """Test scalability with varying client counts."""

    @pytest.mark.parametrize("num_clients", [5, 10, 25, 50, 100])
    def test_client_info_creation_scaling(
        self, num_clients: int, performance_timer: PerformanceTimer
    ) -> None:
        """Test ClientInfo creation time scaling with client count."""
        performance_timer.start()

        # Create multiple ClientInfo objects
        clients: List[ClientInfo] = []
        for client_id in range(num_clients):
            client_info = ClientInfo(client_id=client_id, num_of_rounds=10)

            # Add history data
            for round_num in range(10):
                client_info.add_history_entry(
                    current_round=round_num + 1,  # Rounds are 1-indexed
                    loss=np.random.random(),
                    accuracy=np.random.random(),
                    removal_criterion=np.random.random(),
                )
            clients.append(client_info)

        elapsed_time: float = performance_timer.stop(f"clients_{num_clients}")

        # Time should scale roughly linearly (allow some overhead)
        # Expect less than 0.1 seconds per 10 clients
        expected_max_time: float = (num_clients / 10) * 0.1
        assert (
            elapsed_time < expected_max_time
        ), f"ClientInfo creation too slow for {num_clients} clients: {elapsed_time:.3f}s"

        # Verify all clients were created correctly
        assert len(clients) == num_clients
        for client in clients:
            assert len(client.loss_history) == 10
            assert len(client.accuracy_history) == 10

    @pytest.mark.parametrize("num_clients", [10, 50, 100, 200])
    def test_parameter_aggregation_scaling(
        self, num_clients: int, performance_timer: PerformanceTimer
    ) -> None:
        """Test parameter aggregation time scaling."""
        param_size: int = 10000  # 10K parameters

        # Generate client parameters
        client_params: np.ndarray = generate_mock_client_parameters(
            num_clients, param_size
        )

        performance_timer.start()

        # Simulate parameter aggregation (simple averaging)
        aggregated_params: np.ndarray = np.mean(client_params, axis=0)

        elapsed_time: float = performance_timer.stop(f"aggregation_{num_clients}")

        # Aggregation should be fast and scale linearly
        # Expect less than 0.01 seconds per 50 clients for 10K parameters
        expected_max_time: float = (num_clients / 50) * 0.01
        assert (
            elapsed_time < expected_max_time
        ), f"Parameter aggregation too slow for {num_clients} clients: {elapsed_time:.4f}s"

        # Verify aggregation correctness
        assert aggregated_params.shape == (param_size,)
        assert not np.isnan(aggregated_params).any()

    @pytest.mark.parametrize(
        "num_clients,num_rounds", [(10, 5), (25, 10), (50, 15), (100, 20)]
    )
    def test_simulation_history_scaling(
        self, num_clients: int, num_rounds: int, performance_timer: PerformanceTimer
    ) -> None:
        """Test simulation history creation and access scaling."""
        # Warmup run to stabilize timing
        config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "trust",
                "num_of_rounds": 1,
                "num_of_clients": 5,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            }
        )
        from tests.fixtures.mock_datasets import MockDatasetHandler

        warmup_handler = MockDatasetHandler()
        warmup_handler.setup_dataset(num_clients=5)
        warmup_history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=warmup_handler,
            rounds_history=None,
        )
        warmup_history.insert_round_history_entry(
            score_calculation_time_nanos=100000,
            removal_threshold=0.5,
            loss_aggregated=0.5,
        )

        performance_timer.start()

        # Create mock strategy config
        config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "trust",
                "num_of_rounds": num_rounds,
                "num_of_clients": num_clients,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            }
        )

        # Create mock dataset handler
        dataset_handler = MockDatasetHandler()
        dataset_handler.setup_dataset(num_clients=num_clients)

        # Create history with proper initialization
        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )

        # Create history for multiple rounds
        for round_num in range(num_rounds):
            # Add round history entry
            history.insert_round_history_entry(
                score_calculation_time_nanos=int(np.random.random() * 1000000),
                removal_threshold=np.random.random(),
                loss_aggregated=np.random.random(),
            )

            for client_id in range(num_clients):
                history.insert_single_client_history_entry(
                    client_id=client_id,
                    current_round=round_num + 1,  # Rounds are 1-indexed
                    loss=np.random.random(),
                    accuracy=np.random.random(),
                    removal_criterion=np.random.random(),
                )

        elapsed_time = performance_timer.stop(f"history_{num_clients}x{num_rounds}")

        # History creation should scale reasonably
        expected_max_time = (num_clients * num_rounds) * 0.01
        assert (
            elapsed_time < expected_max_time
        ), f"History creation too slow for {num_clients}x{num_rounds}: {elapsed_time:.4f}s"

        # Verify history structure
        all_clients = history.get_all_clients()
        assert len(all_clients) == num_clients

        # Verify round history
        assert len(history.rounds_history.aggregated_loss_history) == num_rounds


class TestStrategyScalability:
    """Test strategy scalability."""

    @pytest.mark.parametrize(
        "strategy_config",
        [
            {
                "aggregation_strategy_keyword": "trust",
                "num_clients": 20,
                "trust_threshold": 0.7,
            },
            {
                "aggregation_strategy_keyword": "pid",
                "num_clients": 30,
                "Kp": 1.0,
                "Ki": 0.1,
                "Kd": 0.01,
            },
            {
                "aggregation_strategy_keyword": "krum",
                "num_clients": 25,
                "num_krum_selections": 15,
            },
            {
                "aggregation_strategy_keyword": "multi-krum",
                "num_clients": 40,
                "num_krum_selections": 25,
            },
            {
                "aggregation_strategy_keyword": "trimmed_mean",
                "num_clients": 35,
                "trim_ratio": 0.2,
            },
        ],
    )
    def test_strategy_configuration_performance(
        self, strategy_config: Dict[str, Any], performance_timer: PerformanceTimer
    ) -> None:
        """Test strategy configuration performance."""
        # Add common configuration parameters
        full_config = {"num_of_rounds": 5, **strategy_config}

        performance_timer.start()

        # Generate mock client parameters
        num_clients = full_config["num_clients"]
        client_params = generate_mock_client_parameters(num_clients, 5000)

        # Mock strategy execution
        # Create mock strategy instance
        mock_strategy = Mock()
        mock_strategy.aggregate_parameters.return_value = np.random.randn(5000)

        # Simulate strategy processing
        for round_num in range(full_config["num_of_rounds"]):
            _ = mock_strategy.aggregate_parameters(client_params)

        elapsed_time = performance_timer.stop(
            f"strategy_{full_config['aggregation_strategy_keyword']}"
        )

        # Strategy execution should be fast (less than 0.1 seconds for this mock)
        assert (
            elapsed_time < 0.1
        ), f"Strategy {full_config['aggregation_strategy_keyword']} too slow: {elapsed_time:.4f}s"

    @pytest.mark.parametrize("param_size", [1000, 5000, 10000, 50000])
    def test_parameter_size_scaling(
        self, param_size: int, performance_timer: PerformanceTimer
    ) -> None:
        """Test performance scaling with parameter size."""
        num_clients = 20

        # Generate parameters of varying sizes
        client_params = generate_mock_client_parameters(num_clients, param_size)

        performance_timer.start()

        # Test different aggregation operations
        # Simple averaging
        avg_params = np.mean(client_params, axis=0)

        # Weighted averaging (simulate trust-based aggregation)
        weights = np.random.random(num_clients)
        weights = weights / np.sum(weights)
        weighted_params = np.average(client_params, axis=0, weights=weights)

        # Distance calculations (simulate Krum-like operations)
        distances = []
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = np.linalg.norm(client_params[i] - client_params[j])
                distances.append(dist)

        elapsed_time = performance_timer.stop(f"param_size_{param_size}")

        # Time should scale reasonably with parameter size
        # Expect roughly linear scaling (allow some overhead)
        expected_max_time = (param_size / 1000) * 0.01  # 0.01s per 1K parameters
        assert (
            elapsed_time < expected_max_time
        ), f"Parameter operations too slow for size {param_size}: {elapsed_time:.4f}s"

        # Verify results
        assert avg_params.shape == (param_size,)
        assert weighted_params.shape == (param_size,)
        assert len(distances) == (num_clients * (num_clients - 1)) // 2


class TestComputationalComplexity:
    """Test computational complexity bounds for different strategies."""

    def test_trust_strategy_complexity(
        self, performance_timer: PerformanceTimer
    ) -> None:
        """Test computational complexity of trust-based strategy operations."""
        client_counts: List[int] = [10, 20, 40, 80]
        execution_times: List[Tuple[int, float]] = []

        for num_clients in client_counts:
            # Generate client parameters and trust scores
            client_params: np.ndarray = generate_mock_client_parameters(
                num_clients, 1000
            )
            trust_scores: np.ndarray = np.random.uniform(0.3, 1.0, num_clients)

            performance_timer.start()

            # Simulate trust-based aggregation
            # Filter clients based on trust threshold
            threshold: float = 0.7
            trusted_indices: np.ndarray = trust_scores >= threshold
            trusted_params: List[np.ndarray] = [
                client_params[i] for i in range(num_clients) if trusted_indices[i]
            ]

            if trusted_params:
                # Weighted aggregation based on trust scores
                trusted_scores: np.ndarray = trust_scores[trusted_indices]
                weights: np.ndarray = trusted_scores / np.sum(trusted_scores)
                np.average(trusted_params, axis=0, weights=weights)
            else:
                # Fallback to simple averaging
                np.mean(client_params, axis=0)

            elapsed_time: float = performance_timer.stop(f"trust_{num_clients}")
            execution_times.append((num_clients, elapsed_time))

        # Check that complexity is roughly linear
        self._assert_linear_complexity(execution_times, "trust strategy")

    def test_krum_strategy_complexity(
        self, performance_timer: PerformanceTimer
    ) -> None:
        """Test computational complexity of Krum-based strategy operations."""
        client_counts: List[int] = [
            10,
            15,
            20,
            25,
        ]  # Smaller counts due to O(n²) complexity
        execution_times: List[Tuple[int, float]] = []

        for num_clients in client_counts:
            client_params: np.ndarray = generate_mock_client_parameters(
                num_clients, 1000
            )

            performance_timer.start()

            # Simulate Krum distance calculations (O(n²))
            distances = np.zeros((num_clients, num_clients))
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    dist = np.linalg.norm(client_params[i] - client_params[j])
                    distances[i, j] = dist
                    distances[j, i] = dist

            # Find client with minimum sum of distances to closest neighbors
            num_closest = max(1, num_clients - 2)  # f = 2 Byzantine clients
            krum_scores = []
            for i in range(num_clients):
                # Get distances to other clients
                client_distances = distances[i]
                # Sort and take closest neighbors
                closest_distances = np.sort(client_distances)[
                    1 : num_closest + 1
                ]  # Exclude self (distance 0)
                krum_scores.append(np.sum(closest_distances))

            # Select client with minimum Krum score
            selected_client = np.argmin(krum_scores)
            client_params[selected_client]

            elapsed_time = performance_timer.stop(f"krum_{num_clients}")
            execution_times.append((num_clients, elapsed_time))

        # Krum should have quadratic complexity, but still be reasonable for small client counts
        for num_clients, elapsed_time in execution_times:
            # Allow up to 0.01 seconds per client squared (more generous threshold)
            expected_max_time = (num_clients**2) * 0.0001
            assert (
                elapsed_time < expected_max_time
            ), f"Krum strategy too slow for {num_clients} clients: {elapsed_time:.4f}s"

    def test_pid_strategy_complexity(self, performance_timer: PerformanceTimer) -> None:
        """Test computational complexity of PID-based strategy operations."""
        client_counts = [10, 25, 50, 100]
        execution_times = []

        for num_clients in client_counts:
            client_params = generate_mock_client_parameters(num_clients, 1000)

            # Simulate PID controller state
            previous_errors = np.random.randn(num_clients)
            integral_errors = np.random.randn(num_clients)

            performance_timer.start()

            # Simulate PID calculations
            Kp, Ki, Kd = 1.0, 0.1, 0.01

            # Calculate current errors (mock loss values)
            current_errors = np.random.uniform(0.1, 2.0, num_clients)

            # PID calculations
            proportional = Kp * current_errors
            integral_errors += current_errors
            integral = Ki * integral_errors
            derivative = Kd * (current_errors - previous_errors)

            # PID output
            pid_output = proportional + integral + derivative

            # Determine clients to remove based on PID output
            threshold = np.percentile(pid_output, 80)  # Remove top 20%
            keep_indices = pid_output < threshold

            # Aggregate remaining clients
            if np.any(keep_indices):
                kept_params = [
                    client_params[i] for i in range(num_clients) if keep_indices[i]
                ]
                np.mean(kept_params, axis=0)
            else:
                np.mean(client_params, axis=0)

            elapsed_time = performance_timer.stop(f"pid_{num_clients}")
            execution_times.append((num_clients, elapsed_time))

        # Check that complexity is roughly linear
        self._assert_linear_complexity(execution_times, "PID strategy")

    def _assert_linear_complexity(
        self, execution_times: List[Tuple[int, float]], strategy_name: str
    ):
        """Assert that execution times scale roughly linearly with input size."""
        if len(execution_times) < 2:
            return

        # Calculate scaling ratios
        for i in range(1, len(execution_times)):
            prev_clients, prev_time = execution_times[i - 1]
            curr_clients, curr_time = execution_times[i]

            if prev_time > 0:
                client_ratio = curr_clients / prev_clients
                time_ratio = curr_time / prev_time

                # Time ratio should not exceed client ratio by more than 2x (allowing for overhead)
                assert (
                    time_ratio <= client_ratio * 2
                ), f"{strategy_name} complexity issue: {client_ratio:.1f}x clients led to {time_ratio:.1f}x time"


class TestDatasetScalability:
    """Test scalability with different dataset configurations."""

    @pytest.mark.parametrize(
        "dataset_config",
        [
            {
                "num_clients": 10,
                "samples_per_client": 100,
                "input_shape": (1, 28, 28),
            },  # Small
            {
                "num_clients": 25,
                "samples_per_client": 500,
                "input_shape": (3, 32, 32),
            },  # Medium
            {
                "num_clients": 50,
                "samples_per_client": 200,
                "input_shape": (3, 64, 64),
            },  # Large images
            {
                "num_clients": 100,
                "samples_per_client": 100,
                "input_shape": (1, 28, 28),
            },  # Many clients
        ],
    )
    def test_dataset_loading_performance(
        self, dataset_config: Dict[str, Any], performance_timer: PerformanceTimer
    ) -> None:
        """Test dataset loading performance across different configurations."""
        performance_timer.start()

        # Create federated dataset
        fed_dataset = MockFederatedDataset(
            num_clients=dataset_config["num_clients"],
            samples_per_client=dataset_config["samples_per_client"],
            input_shape=dataset_config["input_shape"],
        )

        # Access all client datasets
        for client_id in range(dataset_config["num_clients"]):
            client_dataset = fed_dataset.get_client_dataset(client_id)

            # Sample some data points
            sample_indices = range(
                0, len(client_dataset), max(1, len(client_dataset) // 10)
            )
            for idx in sample_indices:
                _ = client_dataset[idx]

        elapsed_time = performance_timer.stop(
            f"dataset_{dataset_config['num_clients']}clients"
        )

        # Calculate expected time based on total data volume
        total_samples = (
            dataset_config["num_clients"] * dataset_config["samples_per_client"]
        )
        image_size = np.prod(dataset_config["input_shape"])

        # Expect reasonable performance (more generous threshold)
        expected_max_time = (total_samples * image_size) * 0.000001
        expected_max_time = max(expected_max_time, 0.5)  # Minimum 0.5 seconds

        assert (
            elapsed_time < expected_max_time
        ), f"Dataset loading too slow: {elapsed_time:.4f}s for {total_samples} samples"

    @pytest.mark.parametrize("num_rounds", [5, 10, 20, 50])
    def test_multi_round_performance(
        self, num_rounds: int, performance_timer: PerformanceTimer
    ) -> None:
        """Test performance scaling with number of training rounds."""
        num_clients = 20

        # Warmup run to stabilize timing
        from tests.fixtures.mock_datasets import MockDatasetHandler

        warmup_handler = MockDatasetHandler()
        warmup_handler.setup_dataset(num_clients=5)
        generate_mock_client_parameters(5, 100)

        performance_timer.start()

        # Create simulation components
        config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "trust",
                "num_of_rounds": num_rounds,
                "num_of_clients": num_clients,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            }
        )

        # Create mock dataset handler
        dataset_handler = MockDatasetHandler()
        dataset_handler.setup_dataset(num_clients=num_clients)

        # Create history with proper initialization
        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )

        # Simulate multiple rounds
        for round_num in range(num_rounds):
            # Add round history entry
            history.insert_round_history_entry(
                score_calculation_time_nanos=int(np.random.random() * 1000000),
                removal_threshold=np.random.random(),
                loss_aggregated=np.random.random(),
            )

            # Generate client data for this round
            generate_mock_client_parameters(num_clients, 1000)

            for client_id in range(num_clients):
                history.insert_single_client_history_entry(
                    client_id=client_id,
                    current_round=round_num + 1,  # Rounds are 1-indexed
                    loss=np.random.random(),
                    accuracy=np.random.random(),
                    removal_criterion=np.random.random(),
                )

        elapsed_time = performance_timer.stop(f"rounds_{num_rounds}")

        # Time should scale roughly linearly with number of rounds
        expected_max_time = num_rounds * 0.2  # 0.2 seconds per round
        assert (
            elapsed_time < expected_max_time
        ), f"Multi-round simulation too slow for {num_rounds} rounds: {elapsed_time:.4f}s"


class TestByzantineScenarioPerformance:
    """Test performance under Byzantine attack scenarios."""

    @pytest.mark.parametrize(
        "attack_config",
        [
            {"num_clients": 20, "num_byzantine": 2, "attack_type": "gaussian"},
            {"num_clients": 30, "num_byzantine": 5, "attack_type": "zero"},
            {"num_clients": 40, "num_byzantine": 8, "attack_type": "flip"},
            {"num_clients": 50, "num_byzantine": 10, "attack_type": "gaussian"},
        ],
    )
    def test_byzantine_defense_performance(
        self, attack_config: Dict[str, Any], performance_timer: PerformanceTimer
    ) -> None:
        """Test performance of Byzantine defense mechanisms."""
        performance_timer.start()

        # Generate Byzantine client parameters
        client_params = generate_byzantine_client_parameters(
            num_clients=attack_config["num_clients"],
            num_byzantine=attack_config["num_byzantine"],
            param_size=5000,
            attack_type=attack_config["attack_type"],
        )

        # Simulate defense mechanisms
        num_clients = attack_config["num_clients"]

        # Trust-based defense (calculate parameter similarity)
        trust_scores = []
        for i in range(num_clients):
            similarities = []
            for j in range(num_clients):
                if i != j:
                    # Simple cosine similarity
                    dot_product = np.dot(client_params[i], client_params[j])
                    norms = np.linalg.norm(client_params[i]) * np.linalg.norm(
                        client_params[j]
                    )
                    similarity = dot_product / (norms + 1e-8)
                    similarities.append(similarity)
            trust_scores.append(np.mean(similarities))

        # Krum-based defense (distance calculations)
        distances = []
        for i in range(num_clients):
            client_distances = []
            for j in range(num_clients):
                if i != j:
                    dist = np.linalg.norm(client_params[i] - client_params[j])
                    client_distances.append(dist)
            distances.append(
                sorted(client_distances)[: num_clients // 2]
            )  # Closest half

        krum_scores = [sum(dists) for dists in distances]

        # Select honest clients based on defense mechanisms
        trust_threshold = np.percentile(trust_scores, 60)
        krum_threshold = np.percentile(krum_scores, 60)

        honest_by_trust = [
            i for i, score in enumerate(trust_scores) if score >= trust_threshold
        ]
        honest_by_krum = [
            i for i, score in enumerate(krum_scores) if score <= krum_threshold
        ]

        # Aggregate honest clients
        if honest_by_trust:
            honest_params = [client_params[i] for i in honest_by_trust]
            np.mean(honest_params, axis=0)

        elapsed_time = performance_timer.stop(
            f"byzantine_{attack_config['num_clients']}_{attack_config['num_byzantine']}"
        )

        # Byzantine defense should complete in reasonable time
        expected_max_time = (
            attack_config["num_clients"] ** 2
        ) * 0.0001  # Quadratic but with small constant
        assert (
            elapsed_time < expected_max_time
        ), f"Byzantine defense too slow: {elapsed_time:.4f}s for {attack_config['num_clients']} clients"

        # Verify defense effectiveness (should identify some Byzantine clients)
        total_clients = attack_config["num_clients"]

        # At least some clients should be filtered out
        assert (
            len(honest_by_trust) < total_clients
        ), "Trust-based defense should filter some clients"
        assert (
            len(honest_by_krum) < total_clients
        ), "Krum-based defense should filter some clients"


@pytest.mark.slow
class TestLargeScalePerformance:
    """Test performance at larger scales (marked as slow tests)."""

    def test_large_client_simulation(self, performance_timer: PerformanceTimer) -> None:
        """Test simulation performance with large number of clients."""
        num_clients = 500
        num_rounds = 10

        performance_timer.start()

        # Simulate rounds with many clients
        for round_num in range(num_rounds):
            # Generate parameters for all clients
            client_params = generate_mock_client_parameters(num_clients, 1000)

            # Simulate trimmed mean aggregation
            # Sort parameters by magnitude and trim extremes
            param_magnitudes = [np.linalg.norm(params) for params in client_params]
            sorted_indices = np.argsort(param_magnitudes)

            trim_count = int(num_clients * 0.1)
            keep_indices = (
                sorted_indices[trim_count:-trim_count]
                if trim_count > 0
                else sorted_indices
            )

            kept_params = [client_params[i] for i in keep_indices]
            np.mean(kept_params, axis=0)

        elapsed_time = performance_timer.stop(f"large_scale_{num_clients}")

        # Should complete within reasonable time (less than 1 second per round)
        expected_max_time = num_rounds * 1.0
        assert (
            elapsed_time < expected_max_time
        ), f"Large-scale simulation too slow: {elapsed_time:.2f}s for {num_clients} clients"

    def test_large_parameter_aggregation(
        self, performance_timer: PerformanceTimer
    ) -> None:
        """Test aggregation performance with large parameter vectors."""
        num_clients = 100
        param_size = 1_000_000  # 1M parameters (large neural network)

        performance_timer.start()

        # Generate large parameter vectors
        client_params = []
        for client_id in range(num_clients):
            # Use float32 to reduce memory usage
            params = np.random.randn(param_size).astype(np.float32)
            client_params.append(params)

        # Test different aggregation methods
        # Simple averaging
        avg_params = np.mean(client_params, axis=0)

        # Weighted averaging
        weights = np.random.random(num_clients).astype(np.float32)
        weights = weights / np.sum(weights)
        weighted_params = np.average(client_params, axis=0, weights=weights)

        elapsed_time = performance_timer.stop(f"large_params_{param_size}")

        # Should complete within reasonable time (less than 5 seconds)
        assert (
            elapsed_time < 5.0
        ), f"Large parameter aggregation too slow: {elapsed_time:.2f}s for {param_size} parameters"

        # Verify results
        assert avg_params.shape == (param_size,)
        assert weighted_params.shape == (param_size,)
        assert not np.isnan(avg_params).any()
        assert not np.isnan(weighted_params).any()
