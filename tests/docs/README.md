# 🧪 FL Framework Test Suite

Test suite for federated learning simulation framework with multiple aggregation strategies and Byzantine attack scenarios.

## 📋 Overview

Unit tests, integration tests, and performance tests for federated learning components. Tests use synthetic data generation and mocking to validate FL algorithms without requiring real datasets.

### 🎯 Design Principles

- External mocking only - no production code modifications
- Multi-dimensional PyTorch tensors matching real data shapes
- FL-specific scenarios: client heterogeneity, Byzantine attacks, strategy combinations
- Deterministic random seeds for reproducible results
- Parameterized tests across all strategies and datasets

### 📚 Documentation Suite

This `README.md` provides a high-level overview. For detailed information, see:

- **[FL Fundamentals](./fl_fundamentals.md)**: Core federated learning concepts, aggregation strategies, and Byzantine fault tolerance mechanisms
- **[Testing Guide](./testing_guide.md)**: Step-by-step guide for writing, running, and debugging tests
- **[Test Data Generation](./test_data_generation.md)**: Framework for creating mock data and simulating federated learning scenarios
- **[Refactoring for Testability](./refactoring_for_testability.md)**: Technical summary of source code modifications made to enable testing

## 🏗️ Architecture

### 📁 Directory Structure

```text
tests/
├── conftest.py                         # Global fixtures and configuration
├── test_setup.py                      # Test configuration and setup utilities
├── validate_coverage_setup.py          # Test suite setup validation script
├── docs/                              # Test documentation suite
│   ├── README.md                      # This documentation
│   ├── testing_guide.md              # Developer testing guide
│   ├── test_data_generation.md        # Mock data generation guide
│   ├── fl_fundamentals.md             # FL concepts and strategies
│   └── refactoring_for_testability.md # Source code modifications log
├── unit/                             # Unit tests for individual components
│   ├── test_attack_scenarios.py       # Byzantine attack pattern testing
│   ├── test_data_models/              # StrategyConfig, ClientInfo, RoundInfo tests
│   ├── test_config_loaders/           # Configuration parsing and validation tests
│   ├── test_simulation_strategies/    # Aggregation strategy tests
│   ├── test_dataset_handlers/         # Dataset management component tests
│   ├── test_client_models/            # FlowerClient and model interaction tests
│   └── test_network_models/           # Neural network definition tests
├── integration/                       # Multi-component interaction tests
│   ├── test_flower_mock_integration.py # Flower framework integration
│   ├── test_simulation_flow.py        # End-to-end simulation execution
│   ├── test_simulation_runner.py      # Multi-strategy coordination testing
│   └── test_strategy_combinations.py  # Multi-strategy scenario testing
├── performance/                       # Scalability and memory usage tests
│   ├── test_memory_usage.py           # Memory leak detection and monitoring
│   └── test_scalability.py            # Client count and round scaling tests
└── fixtures/                          # Reusable test utilities and mock data
    ├── mock_datasets.py               # Synthetic dataset generation
    ├── mock_flower_components.py      # Flower framework mocks
    ├── sample_models.py               # Lightweight mock network models
    └── __init__.py                    # Fixture package initialization
```

## 🎲 Synthetic Dataset Generation

### 🏛️ Architecture

Synthetic data generation creates federated learning test scenarios without real datasets:

#### 1️⃣ Base MockDataset Class

```python
class MockDataset(Dataset):
    def __init__(self, size=100, num_classes=10, input_shape=(3, 32, 32)):
        torch.manual_seed(42)  # Reproducible generation
        
        # Real PyTorch tensors with proper shapes
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
```

Creates:

- 🔢 Multi-dimensional tensors matching real image data shapes
- 📊 Gaussian-distributed parameters similar to trained neural networks
- ⚙️ PyTorch Dataset interface for DataLoader compatibility

#### 2️⃣ Federated Dataset Simulation

```python
class MockFederatedDataset:
    def _generate_client_datasets(self):
        for client_id in range(self.num_clients):
            # Different seed per client = data heterogeneity
            torch.manual_seed(42 + client_id)
            client_datasets[client_id] = MockDataset(...)
```

Simulates:

- 👥 Client data heterogeneity (Non-IID distribution)
- 🎲 Per-client datasets with different random seeds
- 📈 Variable client populations (5-1000+ clients)

#### 3️⃣ Dataset Type Adaptation

```python
input_shapes = {
    "its": (3, 224, 224),         # RGB traffic sign images
    "femnist_iid": (1, 28, 28),   # Grayscale handwritten characters (IID)
    "femnist_niid": (1, 28, 28),  # Grayscale handwritten characters (Non-IID)
    "pneumoniamnist": (1, 28, 28), # Medical X-ray images
    "bloodmnist": (3, 28, 28),    # Medical blood cell images
    "lung_photos": (3, 224, 224), # High-resolution lung scans (RGB)
    "flair": (3, 224, 224),       # Natural language processing embeddings
}
```

Provides:

- 📐 Dataset-specific tensor dimensions matching production data
- 💾 Memory usage patterns representative of real workloads
- 🏥 Domain-specific characteristics without actual domain data

### ⚡ Advanced Synthetic Features

#### ⚔️ Byzantine Attack Simulation

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

Enables Testing:

- 🛡️ Defense mechanism validation against attack patterns
- ⚔️ Strategy behavior under adversarial conditions
- 🔒 Byzantine fault tolerance across aggregation algorithms

#### 👥 Client Behavior Patterns

```python
# Honest clients - similar parameter updates
if i < honest_count:
    mock_params = [np.random.randn(10, 5) * 0.1, np.random.randn(5) * 0.1]
# Malicious clients - divergent updates
else:
    mock_params = [np.random.randn(10, 5) * (i + 1), np.random.randn(5) * (i + 1)]
```

Tests:

- 🎯 Client selection algorithms (Krum, Multi-Krum, Trust-based)
- ✅ Aggregation correctness under different client behaviors
- 🚫 Threshold-based removal mechanisms

## 📂 Test Categories

### 🔬 Unit Tests

#### ⚔️ Attack Scenarios (`tests/unit/test_attack_scenarios.py`)

Byzantine attack testing across defense strategies:

- Attack Types: Gaussian noise, model poisoning, Byzantine clients, gradient inversion, label flipping, backdoor attacks
- Defense Validation: Trust-based, Krum, Multi-Krum, RFA, Bulyan, Trimmed Mean strategies
- Attack Testing: High, medium, and low attack scenarios with expected outcomes
- Parameterized Testing: All attack-defense combinations tested systematically

#### 📊 Data Models (`tests/unit/test_data_models/`)

- **StrategyConfig**: Initialization, validation, serialization/deserialization
- **ClientInfo**: History management, metric tracking, data integrity
- **RoundInfo**: Round data aggregation and client relationships
- **SimulationStrategyHistory**: Multi-round tracking and consistency

#### ⚙️ Configuration Loading (`tests/unit/test_config_loaders/`)

- **ConfigLoader**: JSON parsing, configuration merging, dataset mapping
- **Strategy Validation**: Parameter validation, error handling, clear error messages

#### 🛡️ Simulation Strategies (`tests/unit/test_simulation_strategies/`)

Multiple aggregation strategies with test coverage:

1. **TrustBasedRemovalStrategy**: Trust score calculation, client removal logic
2. **PIDBasedRemovalStrategy**: PID controller implementation, 3 variants (pid/pid_scaled/pid_standardized)
3. **KrumBasedRemovalStrategy**: Distance calculations, client selection algorithms
4. **MultiKrumBasedRemovalStrategy**: Multi-client selection, consistency validation
5. **TrimmedMeanBasedRemovalStrategy**: Robust averaging, outlier removal
6. **RFABasedRemovalStrategy**: Robust federated averaging implementation
7. **BulyanStrategy**: Byzantine-resistant aggregation with multi-phase selection
8. **Strategy Variations**: Cross-strategy testing and parameterized configurations
9. **Strategy Interactions**: Multi-strategy scenarios and combinations

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

- 📚 Test Infrastructure: Directory structure with fixtures and configuration
- 📐 Data Models: Unit test coverage for data structures and validation logic
- ⚙️ Configuration Management: JSON parsing, validation, error handling with edge cases
- 🛡️ Simulation Strategies: Multiple aggregation algorithms with Byzantine attack scenarios
- 🔗 Strategy Interactions: Multi-strategy combinations and validation
- 🎲 Synthetic Data Generation: Mock dataset infrastructure for FL scenarios
- 📊 Dataset and Client Components: File operations, dataset management, and client model interactions
- 🔗 Integration Testing: End-to-end simulation workflows with component interaction validation
- 📈 Performance Testing: Memory usage monitoring, scalability validation, and computational complexity verification
- ⚙️ CI/CD Integration: Coverage reporting, automated test execution, and quality gates
- 🎯 Parameterized FL Scenarios: Cross-strategy, cross-dataset testing with attack-defense validation

## 🚀 Running Tests

### 🔍 Pre-Test Validation

```bash
# Validate test suite configuration before running tests
python tests/validate_coverage_setup.py

# Expected output on successful validation:
# 🎉 All checks passed (5/5)
# Next steps:
#   1. Run: pytest --cov=src
#   2. Check: htmlcov/index.html
#   3. Commit and push to trigger CI
```

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

# Test attack scenarios
pytest tests/unit/test_attack_scenarios.py -v

# Test strategy variations (multiple strategies)
pytest tests/unit/test_simulation_strategies/test_strategy_variations.py -v
```

## ✨ Key Testing Features

### 🔧 Parameterized Strategy Testing

```python
@pytest.mark.parametrize("strategy_name,config", [
    ("trust", TRUST_CONFIG),
    ("pid", PID_CONFIG),
    ("pid_scaled", PID_SCALED_CONFIG),
    ("pid_standardized", PID_STANDARDIZED_CONFIG), 
    ("krum", KRUM_CONFIG),
    ("multi-krum", MULTI_KRUM_CONFIG),
    ("multi-krum-based", MULTI_KRUM_BASED_CONFIG),
    ("trimmed_mean", TRIMMED_MEAN_CONFIG),
    ("rfa", RFA_CONFIG),
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
    ("femnist_niid", (1, 28, 28)),
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

- 🔢 **Multi-dimensional PyTorch tensors** with proper shapes and distributions
- 👥 **Federated learning specific scenarios** (client heterogeneity, Byzantine attacks)
- ⚖️ **Mathematical correctness validation** using real distance calculations and aggregation algorithms

### 🎯 Coverage

- ✅ **All aggregation strategies** tested individually and in combination
- ⚔️ **Attack-defense scenarios** validating Byzantine fault tolerance
- ⚠️ **Edge cases and error conditions** covered

### 🔧 Maintainable and Scalable

- 🚫 **No modifications to production code** - all testing via external mocks
- 🔄 **Reusable fixtures** and utilities for consistent testing patterns
- 📂 **Clear separation** between unit, integration, and performance tests

This test suite provides realistic tensor operations, proper FL protocol simulation, and validation of distributed algorithm correctness under adversarial conditions.

---

### AI Tool Usage Disclosure

This project utilized generative AI tools (Claude) for development infrastructure and testing framework creation. AI assistance focused on Python best practices, pytest patterns, code quality tooling, and development workflow optimization. All federated learning research, algorithmic implementations, and core technical insights remain original work. AI usage aligns with RIT's responsible AI integration guidelines.
