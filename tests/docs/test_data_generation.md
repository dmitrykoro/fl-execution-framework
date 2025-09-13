# Test Data Generation Framework

## Overview

This document explains the test data generation system for the federated learning simulation framework. The system creates mock datasets, client parameters, and attack scenarios for testing without needing real datasets.

## Core Components

### 1. Mock Dataset Generation (`tests/fixtures/mock_datasets.py`)

#### MockDataset Class

Lightweight dataset replacement that works with PyTorch DataLoaders for testing.

**Technical Features:**

- Configurable input dimensions: `(channels, height, width)`
- Reproducible random data generation with seed control
- Support for multi-class classification scenarios
- Memory-efficient tensor operations

```python
# Example usage
dataset = MockDataset(
    size=100,              # Number of samples
    num_classes=10,        # Classification classes
    input_shape=(3, 32, 32), # RGB 32x32 images
    use_default_seed=True  # Reproducible results
)
```

#### MockFederatedDataset Class

Simulates how data would be split across different federated learning clients.

**Technical Features:**

- Client-specific data partitioning
- Configurable data heterogeneity through different random seeds
- Individual DataLoader generation per client
- Realistic federated learning data scenarios

### 2. Dataset Type Support

#### Supported Configurations

The framework supports 7 different dataset types with appropriate tensor dimensions:

| Dataset Type | Input Shape | Description |
|--------------|-------------|-------------|
| `its` | (3, 224, 224) | RGB high-resolution images |
| `femnist_iid` | (1, 28, 28) | Grayscale MNIST-style (IID) |
| `femnist_niid` | (1, 28, 28) | Grayscale MNIST-style (Non-IID) |
| `flair` | (3, 224, 224) | RGB medical imaging |
| `pneumoniamnist` | (1, 28, 28) | Grayscale medical scans |
| `bloodmnist` | (3, 28, 28) | RGB microscopic images |
| `lung_photos` | (3, 224, 224) | RGB lung X-rays |
| `mock` | (3, 32, 32) | RGB general testing |

### 3. Client Parameter Generation

#### Standard Client Parameters

Generates neural network parameters that simulate what real federated learning clients would send.

```python
def generate_mock_client_parameters(num_clients: int, param_size: int = 1000) -> List[np.ndarray]:
```

**Technical Features:**

- Generates realistic neural network parameter vectors
- Configurable parameter dimensionality
- Gaussian distribution with controlled variance
- Reproducible through seed management

#### Byzantine Attack Simulation

Creates different types of malicious client parameters to test if defense strategies work.

```python
def generate_byzantine_client_parameters(
    num_clients: int,
    num_byzantine: int,
    param_size: int = 1000,
    attack_type: str = "gaussian"
) -> List[np.ndarray]:
```

**Implementation Details:**

1. **Gaussian Noise**: Large-scale random perturbations
2. **Model Poisoning**: Targeted parameter manipulation
3. **Byzantine Clients**: Adversarial parameter corruption  
4. **Gradient Inversion**: Scaled parameter attacks
5. **Label Flipping**: Sign-flipped parameters simulating mislabeling
6. **Backdoor Attack**: Fixed pattern injection for model compromise
7. **Zero Attack**: All-zero parameter submission
8. **Flip Attack**: Negated honest parameters

### 4. Client Metrics Generation

#### Historical Metrics Simulation

Generates fake client performance metrics (loss, accuracy, F1-score) for testing.

```python
def generate_mock_client_metrics(num_clients: int, num_rounds: int) -> Dict[int, Dict[str, List[float]]]:
```

**Technical Features:**

- **Loss**: Range [0.1, 2.0] simulating training convergence patterns
- **Accuracy**: Range [0.5, 0.95] representing realistic model performance trajectories
- **F1-Score**: Range [0.4, 0.9] for classification quality assessment

## Integration with Testing Framework

### Pytest Framework Integration

Provides shared test configurations and mock components that all tests can use.

**Technical Features:**

- **Global Configuration**: Strategy-specific parameter sets for all test scenarios
- **Mock Components**: Network models, client handlers, and dataset managers
- **Temporary Resources**: File system mocking and automated cleanup
- **Parameterized Testing**: Full support for all 10 aggregation strategies
- **Attack Scenario Testing**: Comprehensive Byzantine attack pattern validation

#### Test Coverage Areas

Tests different aspects of the framework to make sure everything works correctly.

**Implementation Details:**

1. **Unit Tests**: Individual component validation with controlled inputs
2. **Integration Tests**: Multi-component workflow verification and end-to-end testing
3. **Performance Tests**: Memory usage monitoring and scalability assessment
4. **Attack Scenario Tests**: Byzantine resilience validation under adversarial conditions
5. **Dataset Variation Tests**: Cross-dataset compatibility verification and adaptation

## Technical Implementation Details

### Tensor Dimension Management

- **Dynamic Sizing**: Automatic adaptation to different dataset requirements
- **Memory Optimization**: Efficient tensor allocation and deallocation
- **GPU Compatibility**: CUDA-aware tensor generation when available
- **Batch Processing**: Support for configurable batch sizes across all datasets

### Reproducibility Features

- **Seed Control**: Deterministic random generation across test runs
- **Parameter Consistency**: Identical outputs for identical configurations
- **Cross-Platform Compatibility**: Windows/Linux/macOS consistent behavior
- **Version Independence**: Stable across PyTorch and NumPy versions

### Performance Considerations

- **Lazy Loading**: On-demand dataset generation to minimize memory usage
- **Vectorized Operations**: NumPy/PyTorch optimized computations
- **Scalable Architecture**: Efficient handling from 5-1000+ clients
- **Resource Cleanup**: Automatic memory management and garbage collection

## Usage Examples

### Basic Strategy Testing

```python
# Generate test data for trust-based strategy
client_params = generate_mock_client_parameters(num_clients=10, param_size=1000)
federated_dataset = MockFederatedDataset(num_clients=10, input_shape=(3, 32, 32))

# Test aggregation with mock data
strategy = TrustBasedRemovalStrategy(trust_threshold=0.7)
result = strategy.aggregate_fit(client_params)
```

### Byzantine Attack Testing  

```python
# Generate Byzantine client parameters
byzantine_params = generate_byzantine_client_parameters(
    num_clients=10,
    num_byzantine=3,
    attack_type="model_poisoning"
)

# Test defense mechanism effectiveness
robust_strategy = BulyanStrategy(num_byzantine=3)
aggregated = robust_strategy.aggregate_fit(byzantine_params)
```

### Dataset Variation Testing

```python
# Test across all supported dataset types
dataset_types = ["its", "femnist_iid", "femnist_niid", "flair", "pneumoniamnist", "bloodmnist", "lung_photos"]
for dataset_type in dataset_types:
    handler = MockDatasetHandler(dataset_type=dataset_type)
    handler.setup_dataset(num_clients=10)
    # Run simulation with dataset-specific dimensions
```

## Quality Assurance

### Validation Mechanisms

- **Dimension Verification**: Automatic tensor shape validation
- **Range Checking**: Parameter value bounds enforcement  
- **Statistical Properties**: Distribution characteristic validation
- **Attack Effectiveness**: Byzantine attack success rate measurement

### Testing Standards

- **Code Coverage**: >70% line coverage across all mock components
- **Performance Benchmarks**: Sub-second generation for standard test cases
- **Memory Efficiency**: <100MB peak usage for typical test scenarios
- **Error Handling**: Comprehensive exception testing and recovery

## Future Enhancements

### Planned Features

- **Advanced Attack Patterns**: More sophisticated Byzantine attack simulations
- **Realistic Data Distributions**: Non-uniform client data partitioning
- **Dynamic Client Participation**: Simulated client dropouts and reconnections
- **Hardware Simulation**: Different client computational capability modeling

## Summary

This test data generation system enables testing the federated learning framework without needing real datasets. The system provides:

- **Dataset Support**: 7 different dataset types with correct tensor dimensions  
- **Attack Testing**: 8 different attack patterns to test defense strategies
- **Scalable Testing**: Works with few clients or many clients
- **Pytest Integration**: Works seamlessly with the pytest testing framework

The framework allows thorough testing of federated learning algorithms without requiring real-world data, making development and research much easier.
