# 🧪 Federated Learning PyTest Test Suite

A testing framework for the federated learning simulation codebase, featuring synthetic data generation, mocking strategies, and coverage of FL-specific scenarios.

## Overview

This test suite provides unit tests, integration tests, and performance benchmarks for a federated learning framework that implements multiple aggregation strategies and Byzantine fault tolerance mechanisms. The testing approach prioritizes minimal changes to production code while validating critical FL components.

### Key Design Principles

- **No Production Code Modifications**: All tests use external mocking without touching source code
- **Realistic Synthetic Data**: Multi-dimensional PyTorch tensors, not simple integers
- **FL-Specific Testing**: Client heterogeneity, Byzantine attacks, and strategy interactions
- **Reproducible Results**: Deterministic random seeds for consistent test outcomes
- **Scalable Architecture**: Parameterized tests across strategies, datasets, and attack scenarios

## 🏗️ Architecture

### Directory Structure

```text
tests/
├── README.md                           # This documentation
├── conftest.py                         # Global fixtures and configuration
├── pytest.ini                         # PyTest execution configuration
├── unit/                              # Unit tests for individual components
│   ├── test_data_models/              # StrategyConfig, ClientInfo, RoundInfo tests
│   ├── test_config_loaders/           # Configuration parsing and validation tests
│   ├── test_simulation_strategies/    # All 9 aggregation strategy tests
│   ├── test_dataset_handlers/         # Dataset management component tests
│   ├── test_client_models/            # FlowerClient and model interaction tests
│   └── test_network_models/           # Neural network definition tests
├── integration/                       # Multi-component interaction tests
│   ├── test_simulation_flow.py        # End-to-end simulation execution
│   ├── test_simulation_runner.py      # Multi-strategy coordination testing
│   └── test_strategy_combinations.py  # Multi-strategy scenario testing
├── performance/                       # Scalability and memory usage tests
│   ├── test_memory_usage.py           # Memory leak detection and monitoring
│   └── test_scalability.py            # Client count and round scaling tests
└── fixtures/                         # Reusable test utilities and mock data
    ├── mock_datasets.py              # Synthetic dataset generation
    └── sample_models.py              # Lightweight mock network models
```

## 🎲 Synthetic Dataset Generation

### Multi-Layered Architecture

Our synthetic data generation creates **realistic federated learning scenarios** without depending on actual datasets:

#### 1. Base MockDataset Class

```python
class MockDataset(Dataset):
    def __init__(self, size=100, num_classes=10, input_shape=(3, 32, 32)):
        torch.manual_seed(42)  # Reproducible generation
        
        # Real PyTorch tensors with proper shapes
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
```

**Creates:**

- **Multi-dimensional tensors** matching real image data shapes
- **Gaussian-distributed parameters** similar to trained neural networks
- **Proper PyTorch Dataset interface** for seamless integration

#### 2. Federated Dataset Simulation

```python
class MockFederatedDataset:
    def _generate_client_datasets(self):
        for client_id in range(self.num_clients):
            # Different seed per client = data heterogeneity
            torch.manual_seed(42 + client_id)
            client_datasets[client_id] = MockDataset(...)
```

**Simulates:**

- **Client data heterogeneity** (Non-IID distribution)
- **Realistic FL scenarios** with per-client datasets
- **Scalable client populations** for testing different federation sizes

#### 3. Dataset Type Adaptation

```python
input_shapes = {
    "its": (3, 224, 224),         # RGB traffic sign images
    "femnist_iid": (1, 28, 28),   # Grayscale handwritten characters
    "pneumoniamnist": (1, 28, 28), # Medical X-ray images
    "bloodmnist": (3, 28, 28),    # Medical blood cell images
    "lung_photos": (1, 224, 224), # High-resolution lung scans (grayscale)
    "flair": (3, 224, 224),       # Natural language processing embeddings
}
```

**Provides:**

- **Dataset-specific tensor dimensions** matching production data
- **Memory usage patterns** representative of real workloads
- **Domain-specific characteristics** without actual domain data

### Advanced Synthetic Features

#### Byzantine Attack Simulation

```python
def generate_byzantine_client_parameters(num_clients, num_byzantine, attack_type):
    if attack_type == "gaussian":
        # Large Gaussian noise injection
        byzantine_params.append(np.random.randn(param_size) * 10)
    elif attack_type == "zero":
        # Zero gradient attack
        byzantine_params.append(np.zeros(param_size))
    elif attack_type == "flip":
        # Sign flipping attack
        byzantine_params.append(-base_param * 5)
```

**Enables Testing:**

- **Defense mechanism validation** against various attack patterns
- **Strategy robustness evaluation** under adversarial conditions
- **Byzantine fault tolerance** across different aggregation algorithms

#### Client Behavior Patterns

```python
# Honest clients - similar parameter updates
if i < honest_count:
    mock_params = [np.random.randn(10, 5) * 0.1, np.random.randn(5) * 0.1]
# Malicious clients - divergent updates
else:
    mock_params = [np.random.randn(10, 5) * (i + 1), np.random.randn(5) * (i + 1)]
```

**Tests:**

- **Client selection algorithms** (Krum, Multi-Krum, Trust-based)
- **Aggregation correctness** under different client behaviors  
- **Threshold-based removal** mechanisms

## 📂 Test Categories

### 🔬 Unit Tests

#### 📊 Data Models (`tests/unit/test_data_models/`)

- **StrategyConfig**: Initialization, validation, serialization/deserialization
- **ClientInfo**: History management, metric tracking, data integrity
- **RoundInfo**: Round data aggregation and client relationships
- **SimulationStrategyHistory**: Multi-round tracking and consistency

#### ⚙️ Configuration Loading (`tests/unit/test_config_loaders/`)

- **ConfigLoader**: JSON parsing, configuration merging, dataset mapping
- **Strategy Validation**: Parameter validation, error handling, clear error messages

#### 🛡️ Simulation Strategies (`tests/unit/test_simulation_strategies/`)

All 9 aggregation strategies with test coverage:

1. **TrustBasedRemovalStrategy**: Trust score calculation, client removal logic
2. **PIDBasedRemovalStrategy**: PID controller implementation, 3 variants (pid/pid_scaled/pid_standardized)
3. **KrumBasedRemovalStrategy**: Distance calculations, client selection algorithms  
4. **MultiKrumBasedRemovalStrategy**: Multi-client selection, consistency validation
5. **TrimmedMeanBasedRemovalStrategy**: Robust averaging, outlier removal
6. **RFABasedRemovalStrategy**: Robust federated averaging implementation
7. **BulyanStrategy**: Byzantine-robust aggregation with multi-phase selection
8. **MultiKrumStrategy**: Multi-Krum aggregation without removal mechanisms
9. **Strategy Interactions**: Complex multi-strategy scenarios and combinations

#### 💾 Dataset and Client Components (`tests/unit/test_dataset_handlers/`, `tests/unit/test_client_models/`)

- **DatasetHandler**: Dataset setup/teardown, file operations, configuration handling
- **FlowerClient**: Model operations, training/evaluation with mocked PyTorch components
- **Network Models**: Architecture initialization, parameter extraction, state management

### 🔗 Integration Tests

#### ⚡ Simulation Workflows (`tests/integration/`)

- **FederatedSimulation**: End-to-end execution with mocked Flower components
- **SimulationRunner**: Multi-strategy coordination and parameter inheritance
- **Strategy Combinations**: Complex multi-strategy interaction testing

### 📊 Performance Tests

#### 📈 Scalability and Resource Management (`tests/performance/`)

- **Memory Usage**: Memory consumption monitoring and leak detection
- **Scalability**: Client count and round scaling performance characteristics
- **Computational Complexity**: Algorithm efficiency validation across strategies

## 🎭 Mocking Strategy

### 🌸 Flower FL Component Mocking

```python
class MockFlowerClient:
    def fit(self, parameters, config):
        # Return realistic mock parameters and metrics
        return mock_parameters, len(self.mock_data), {"loss": 0.1}
    
    def evaluate(self, parameters, config):
        # Return mock evaluation results
        return 0.1, len(self.mock_data), {"accuracy": 0.9}
```

### 🔥 PyTorch Operation Mocking

```python
class MockNetwork(nn.Module):
    def __init__(self, num_classes=10, input_size=3072):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)  # Lightweight architecture
        
    def get_parameters(self):
        return [param.detach().numpy() for param in self.parameters()]
```

### 📁 File I/O and Output Mocking

```python
@pytest.fixture
def mock_output_directory(tmp_path, monkeypatch):
    """Create proper output directory structure for tests."""
    output_dir = tmp_path / "out" / "test_run"
    output_dir.mkdir(parents=True)
    (output_dir / "output.log").touch()  # Create expected log file
    
    # Mock DirectoryHandler.dirname to point to our test directory
    monkeypatch.setattr('src.output_handlers.directory_handler.DirectoryHandler.dirname', str(output_dir))
    
    return output_dir
```

## 🏛️ Implementation Architecture

### ⚡ Core Components

- **Test Infrastructure**: Complete directory structure with organized fixtures and configuration
- **Data Models**: Unit test coverage for all data structures and validation logic
- **Configuration Management**: JSON parsing, validation, error handling with edge case coverage
- **Simulation Strategies**: All 9 aggregation algorithms with extensive testing including Byzantine attack scenarios
- **Strategy Interactions**: Multi-strategy combinations and robustness validation
- **Synthetic Data Generation**: Mock dataset infrastructure with realistic FL scenarios
- **Dataset and Client Components**: Complete testing of file operations, dataset management, and client model interactions
- **Integration Testing**: End-to-end simulation workflows with component interaction validation
- **Performance Testing**: Memory usage monitoring, scalability validation, and computational complexity verification
- **CI/CD Integration**: Coverage reporting, automated test execution, and quality gate enforcement
- **Parameterized FL Scenarios**: Cross-strategy, cross-dataset testing with attack-defense validation

## 🚀 Running Tests

### ⚡ Basic Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test category
pytest tests/unit/test_simulation_strategies/ -v

# Run single test file
pytest tests/unit/test_data_models/test_strategy_config.py -v
```

### 📊 Coverage Reporting

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# Coverage with minimum threshold
pytest --cov=src --cov-fail-under=70
```

### 🔍 Test Filtering

```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m performance
```

### 🎯 Specific Strategy Testing

```bash
# Test specific aggregation strategy
pytest tests/unit/test_simulation_strategies/test_trust_strategy.py -v

# Test strategy interactions
pytest tests/unit/test_simulation_strategies/test_strategy_interactions.py -v
```

## ✨ Key Testing Features

### 🔧 Parameterized Strategy Testing

```python
@pytest.mark.parametrize("strategy_name,config", [
    ("trust", TRUST_CONFIG),
    ("pid", PID_CONFIG), 
    ("krum", KRUM_CONFIG),
    ("multi-krum", MULTI_KRUM_CONFIG),
    ("bulyan", BULYAN_CONFIG),
])
def test_strategy_execution(strategy_name, config):
    # Test each strategy with specific configuration
```

### ⚔️ Attack Scenario Validation

```python
@pytest.mark.parametrize("attack_type,defense_strategies", [
    ("gaussian_noise", ["trust", "krum", "rfa"]),
    ("model_poisoning", ["multi-krum", "bulyan", "trimmed_mean"]),
    ("byzantine_clients", ["trust", "krum", "rfa", "bulyan"]),
])
def test_defense_mechanisms(attack_type, defense_strategies):
    # Validate defense effectiveness against specific attacks
```

### 📋 Dataset Variation Testing  

```python
@pytest.mark.parametrize("dataset_type,expected_shape", [
    ("its", (3, 224, 224)),
    ("femnist_iid", (1, 28, 28)),
    ("pneumoniamnist", (1, 28, 28)),
    ("lung_photos", (3, 224, 224)),
])
def test_dataset_compatibility(dataset_type, expected_shape):
    # Ensure strategies work across different data types
```

## 📝 Test Development Guidelines

### 🗂️ Test Structure and Organization

The test suite follows a systematic approach to ensure coverage and maintainability:

1. **Directory Organization**: Tests are organized by component type with clear separation between unit, integration, and performance tests
2. **Fixture Utilization**: Fixtures from `conftest.py` provide consistent, reusable test data and mock objects
3. **Reproducible Results**: Deterministic random seeds ensure consistent test outcomes across different environments
4. **Mocking**: External dependencies are properly mocked without modifying production code

### 🎯 Mock Data Conventions

- **Deterministic Generation**: `torch.manual_seed(42)` ensures reproducible tensor generation
- **Realistic Tensor Shapes**: Mock data matches actual component requirements and real-world data dimensions
- **Edge Case Coverage**: Tests include empty inputs, single clients, and extreme parameter variations
- **FL Scenario Simulation**: Mock data simulates client heterogeneity and Byzantine attack patterns

### 🏷️ Fixture Naming Patterns

- `mock_*`: Mock objects and synthetic data generators
- `sample_*`: Sample configurations and representative test data
- `temp_*`: Temporary files and directories for isolated testing
- `strategy_*`: Strategy-specific fixtures and configurations

## ✅ Why This Approach Works

### 🎪 Realistic Testing Without Real Data

- **Multi-dimensional PyTorch tensors** with proper shapes and distributions
- **Federated learning specific scenarios** (client heterogeneity, Byzantine attacks)
- **Mathematical correctness validation** using real distance calculations and aggregation algorithms

### 🎯 Coverage

- **All aggregation strategies** tested individually and in combination
- **Attack-defense scenarios** validating Byzantine fault tolerance
- **Edge cases and error conditions** covered

### 🔧 Maintainable and Scalable

- **No modifications to production code** - all testing via external mocks
- **Reusable fixtures** and utilities for consistent testing patterns
- **Clear separation** between unit, integration, and performance tests

This test suite demonstrates that federated learning testing **goes beyond simple integers** - it requires realistic tensor operations, proper FL protocol simulation, and validation of distributed algorithm correctness under adversarial conditions.
