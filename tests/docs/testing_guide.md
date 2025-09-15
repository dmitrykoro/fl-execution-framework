# 👨‍💻 Developer Testing Guide

## 📑 Table of Contents

### 🚀 Getting Started

- [Quick Command Reference](#-quick-command-reference-copy--paste-these)
- [Quick Start for New Developers](#-quick-start-for-new-developers)
- [Your First Test (Step-by-step walkthrough)](#-your-first-test-step-by-step-walkthrough)
- [Learning Path for New Researchers](#-learning-path-for-new-researchers)

### 📚 Testing Fundamentals

- [Testing Fundamentals](#-testing-fundamentals)
- [The AAA Pattern (Your New Best Friend)](#-the-aaa-pattern-your-new-best-friend)
- [Understanding Assertions (The Heart of Testing)](#️-understanding-assertions-the-heart-of-testing)
- [Mock Data: Why We Don't Use Real Data](#-mock-data-why-we-dont-use-real-data)
- [Test Classes: Organizing Your Tests](#-test-classes-organizing-your-tests)
- [Fixtures: Reusable Test Data](#-fixtures-reusable-test-data)

### ➕ Adding Tests

- [Adding Tests for New Source Code](#-adding-tests-for-new-source-code)
- [Adding Tests for LLM-Generated Code](#-adding-tests-for-llm-generated-code)

### 🏃 Running & Debugging

- [Running Your Tests](#-running-your-tests)
- [Testing Best Practices](#-testing-best-practices)
- [Common Beginner Mistakes](#-common-beginner-mistakes)
- [Common Testing Pitfalls](#️-common-testing-pitfalls)
- [Troubleshooting Common Issues](#-troubleshooting-common-issues)
- [Getting Help](#-getting-help)

### ✅ Final Steps

- [Checklist for New Researchers](#-checklist-for-new-researchers)

---

## ⚡ Quick Command Reference (Copy & Paste These!)

```bash
# Core commands (use "python3" if "python" doesn't work):
python tests/validate_coverage_setup.py                          # Validate test suite setup
python -m pytest --version                                       # Check pytest works
python -m pytest tests/unit/test_data_models/ --no-cov -v       # Run unit tests
pwd                                                               # Verify you're in project root
python -m pytest tests/unit/test_my_first_test.py --no-cov -v -s # Run tutorial test (after creating it)
```

## 🚀 Quick Start for New Developers

### 📎 Prerequisites

1. **Environment Setup**: Python 3.10+ and virtual environment activated
2. **Test Suite Validation**: Run `python tests/validate_coverage_setup.py` to verify setup
3. **Test Familiarity**: Basic knowledge of pytest and Python testing
4. **Framework Understanding**: Read `tests/docs/README.md` for architecture overview

### 🎆 Your First Test (Step-by-step walkthrough)

**🎯 Goal**: Create and run your first test to understand the framework

#### 1️⃣ Verify Your Environment

```bash
# Make sure you're in the project root directory
pwd  # Should show: /path/to/fl-execution-framework

# Activate your virtual environment (if not already active)
source .venv/Scripts/activate  # Windows Git Bash
# OR
source .venv/bin/activate      # Linux/Mac

# Validate test suite setup
python tests/validate_coverage_setup.py

# Verify pytest works
python -m pytest --version
```

#### 2️⃣ Run Existing Tests (Learn by Example)

```bash
# Run a simple test to see what "passing tests" look like
python -m pytest tests/unit/test_data_models/test_strategy_config.py -v

# You should see output like:
# ✅ test_initialization_valid_parameters PASSED
# ✅ test_from_dict_creates_valid_config PASSED
```

#### 3️⃣ Create Your Test File

```bash
# Navigate to the unit test directory
cd tests/unit

# Create your test file (MUST start with "test_")
touch test_my_first_test.py

# Go back to project root
cd ../..
```

#### 4️⃣ Write Your First Test

Open `tests/unit/test_my_first_test.py` and copy this code:

```python
"""My first test - learning the framework step by step."""

import pytest
from tests.conftest import generate_mock_client_data


class TestMyFirstExperience:
    """Learning how to write tests for federated learning."""

    def test_framework_is_working(self):
        """Test 1: Verify I can generate mock data."""
        # STEP 1: Generate some fake client data
        client_results = generate_mock_client_data(num_clients=3)

        # STEP 2: Check what we got (client_results contains (ClientProxy, FitRes) tuples)
        print(f"\n🔍 Generated data for {len(client_results)} clients")
        print(f"🔍 First client ID: {client_results[0][0].cid}")
        print(f"🔍 Client has FitRes with parameters and metrics")

        # STEP 3: Make assertions (these MUST be true for test to pass)
        assert len(client_results) == 3, "Should have exactly 3 clients"
        assert hasattr(client_results[0][1], 'parameters'), "Should have parameters"
        assert hasattr(client_results[0][1], 'num_examples'), "Should have num_examples"

        print("✅ Test passed! Framework is working correctly.")

    def test_understanding_assertions(self):
        """Test 2: Learn how assertions work in testing."""
        # Assertions are the CORE of testing - they check if something is true

        # This will PASS (True assertion)
        assert 2 + 2 == 4, "Math should work correctly"

        # This will PASS (checking data types)
        my_list = [1, 2, 3]
        assert isinstance(my_list, list), "Should be a list"
        assert len(my_list) == 3, "Should have 3 elements"

        # This will PASS (checking conditions)
        for item in my_list:
            assert item > 0, f"Item {item} should be positive"

        print("✅ I understand how assertions work!")

    def test_learning_mock_data(self):
        """Test 3: Understand what mock data looks like."""
        # Generate mock data for federated learning
        client_results = generate_mock_client_data(num_clients=2)

        # Let's inspect the data structure (client_results = [(ClientProxy, FitRes), ...])
        client_proxy_1, fit_res_1 = client_results[0]  # First client's data
        client_proxy_2, fit_res_2 = client_results[1]  # Second client's data

        print(f"\n📊 Client 1 ID: {client_proxy_1.cid}")
        print(f"📊 Client 1 examples: {fit_res_1.num_examples}")
        print(f"📊 Client 1 has metrics: {fit_res_1.metrics}")

        # Key understanding: Each result contains ClientProxy and FitRes
        # FitRes has parameters, num_examples, and metrics
        assert hasattr(fit_res_1, 'parameters'), "Should have parameters"
        assert hasattr(fit_res_1, 'num_examples'), "Should have num_examples"
        assert hasattr(fit_res_1, 'metrics'), "Should have metrics"

        print("✅ I understand the mock data structure!")
```

#### 5️⃣ Run Your Test

```bash
# Run your specific test file with verbose output and print statements (bypassing coverage)
python -m pytest tests/unit/test_my_first_test.py -v -s --no-cov

# You should see:
# test_framework_is_working PASSED ✅
# test_understanding_assertions PASSED ✅
# test_learning_mock_data PASSED ✅
```

#### 6️⃣ Understanding Test Output

**✅ SUCCESS - You'll see this when everything works:**

```bash
tests/unit/test_my_first_test.py::TestMyFirstExperience::test_framework_is_working PASSED
tests/unit/test_my_first_test.py::TestMyFirstExperience::test_understanding_assertions PASSED
tests/unit/test_my_first_test.py::TestMyFirstExperience::test_learning_mock_data PASSED

🔍 Generated data for 3 clients
✅ Test passed! Framework is working correctly.

========================= 3 passed in 0.12s =========================
```

**❌ FAILURE - Common issues and fixes:**

```bash
FAILED tests/unit/test_my_first_test.py::TestMyFirstExperience::test_framework_is_working

ImportError: No module named 'tests.fixtures.mock_datasets'
```

**Fix:** Make sure you're in the project root directory and virtual environment is active

## Overview

This guide helps student researchers add tests when extending the federated learning framework. Follow these patterns to maintain code quality and validate your contributions.

## 📚 Learning Path for New Researchers

### 🔍 Phase 1: Understanding Existing Tests

**🎯 Goal**: Learn how tests work in this framework

**Step-by-step approach**:

1. **Read this guide** - Understand testing patterns
2. **Run and study existing tests**:

   ```bash
   # Run a simple test and watch what happens
   python -m pytest tests/unit/test_data_models/test_strategy_config.py -v -s

   # Open the test file and read the code
   cat tests/unit/test_data_models/test_strategy_config.py
   ```

3. **Understand the patterns**:
   - How tests are organized in classes
   - How `assert` statements work
   - What mock data looks like
4. **Explore the fixtures**:

   ```bash
   # Look at how mock data is generated
   cat tests/fixtures/mock_datasets.py
   ```

### ✍️ Phase 2: Adding Simple Tests

**🎯 Goal**: Write your first real test for the framework

**Practical exercises**:

1. **Start with data models** (they're the easiest):

   ```bash
   # Study an existing data model test
   python -m pytest tests/unit/test_data_models/test_client_info.py -v -s
   ```

2. **Practice the AAA pattern**:
   - **Arrange**: Set up your test data
   - **Act**: Call the function you're testing
   - **Assert**: Check that it worked correctly
3. **Master edge cases**:
   - What happens with empty inputs?
   - What happens with invalid data?
   - What happens at boundary conditions?

### 🛡️ Phase 3: Strategy Testing

**🎯 Goal**: Test federated learning algorithms

**Advanced concepts**:

1. **Understand aggregation algorithms** - These are the heart of FL
2. **Learn Byzantine attack patterns** - How to test attack resistance
3. **Test complex scenarios** - Multiple clients, different attack types

## 🧠 Testing Fundamentals

### ❓ What is Testing?

**Testing is asking the question**: "Does my code actually work the way I think it does?"

Instead of manually running your function and checking output, you write **automated checks** that verify correctness.

#### 🧠 The Testing Mindset

```python
# Instead of this (manual testing):
def my_function(x):
    return x * 2

result = my_function(5)  # I look at result and think "yep, 10 is correct"

# We write this (automated testing):
def test_my_function():
    result = my_function(5)
    assert result == 10, f"Expected 10, but got {result}"
```

### 🔄 The AAA Pattern (Your New Best Friend)

Every good test follows this structure:

#### 1️⃣ **ARRANGE** (Set up your test data)

```python
def test_client_aggregation():
    # ARRANGE: Create the data you need for testing
    client_params = generate_mock_client_parameters(num_clients=3, param_size=100)
    strategy = TrustBasedRemovalStrategy(trust_threshold=0.5)
```

#### 2️⃣ **ACT** (Call the function you're testing)

```python
    # ACT: Call the function you want to test
    result = strategy.aggregate_fit(
        server_round=1,
        results=client_params,
        failures=[]
    )
```

#### 3️⃣ **ASSERT** (Check that it worked correctly)

```python
    # ASSERT: Verify the result is what you expected
    assert result is not None, "Should return aggregated parameters"
    assert len(result) > 0, "Should have non-empty result"
```

### ❤️ Understanding Assertions (The Heart of Testing)

An assertion is a statement that **must be true** for your test to pass.

```python
# Basic assertion patterns:

# Equality checks
assert actual == expected, "Values should be equal"

# Type checks
assert isinstance(result, list), "Result should be a list"

# Length/size checks
assert len(data) == 5, "Should have exactly 5 items"

# Presence checks
assert "key" in dictionary, "Key should exist in dictionary"

# Range checks
assert 0 <= value <= 1, "Value should be between 0 and 1"

# Exception testing (for error cases)
with pytest.raises(ValueError):
    my_function(invalid_input)  # Should raise ValueError
```

### 🎭 Mock Data: Why We Don't Use Real Data

**Problem**: Real federated learning data is:

- Huge (gigabytes)
- Slow to process
- Inconsistent (different shapes, formats)
- Private (can't include in code)

**Solution**: Create **fake data** that has the same structure:

```python
# Instead of loading real MNIST data (slow):
real_data = load_mnist()  # Takes 30 seconds, 60MB download

# We generate fake data (fast):
fake_data = generate_mock_client_parameters(3, 100)  # Takes 0.001 seconds

# The fake data has the SAME STRUCTURE as real data:
# - Same array shapes
# - Same data types
# - Same ranges
# But it's completely artificial and fast to generate
```

### 📁 Test Classes: Organizing Your Tests

```python
class TestMyFeature:
    """Group related tests together in a class."""

    def test_normal_case(self):
        """Test the typical usage."""
        pass

    def test_edge_case_empty_input(self):
        """Test what happens with empty input."""
        pass

    def test_error_case_invalid_input(self):
        """Test what happens with bad input."""
        pass
```

**Why use classes?**

- **Organization**: Keep related tests together
- **Setup sharing**: Common setup code for all tests in the class
- **Clear naming**: `TestStrategyAggregation` tells you what's being tested

### 🔧 Fixtures: Reusable Test Data

Instead of creating the same test data in every test:

```python
# Bad: Repetitive setup in each test
def test_aggregation_1():
    client_data = generate_mock_client_parameters(3, 100)  # Repeated
    strategy = TrustBasedRemovalStrategy()  # Repeated
    # ... test code

def test_aggregation_2():
    client_data = generate_mock_client_parameters(3, 100)  # Repeated
    strategy = TrustBasedRemovalStrategy()  # Repeated
    # ... test code
```

```python
# Good: Use fixtures (shared setup)
@pytest.fixture
def sample_client_data():
    return generate_mock_client_parameters(3, 100)

@pytest.fixture
def trust_strategy():
    return TrustBasedRemovalStrategy()

def test_aggregation_1(sample_client_data, trust_strategy):
    # Data is automatically provided by fixtures
    result = trust_strategy.aggregate_fit(1, sample_client_data, [])

def test_aggregation_2(sample_client_data, trust_strategy):
    # Same fixtures, no repetition
    result = trust_strategy.aggregate_fit(2, sample_client_data, [])
```

## ➕ Adding Tests for New Source Code

### 1️⃣ New Strategy Implementation

When you add a new aggregation strategy to `src/simulation_strategies/`:

**Step 1**: Create the strategy test file

```bash
# Create test file following naming convention
touch tests/unit/test_simulation_strategies/test_your_strategy.py
```

**Step 2**: Use the standard strategy test template

```python
"""Unit tests for YourStrategy aggregation algorithm."""

import numpy as np
import pytest
from unittest.mock import Mock

from src.simulation_strategies.your_strategy import YourStrategy
from tests.fixtures.mock_datasets import generate_mock_client_parameters


class TestYourStrategy:
    """Test suite for YourStrategy aggregation."""

    def test_initialization(self):
        """Test strategy initialization with valid parameters."""
        strategy = YourStrategy(
            your_param=1.0,
            begin_removing_from_round=2
        )
        assert strategy.your_param == 1.0
        assert strategy.begin_removing_from_round == 2

    def test_aggregate_fit_normal_case(self):
        """Test aggregation with typical client parameters."""
        strategy = YourStrategy(your_param=0.5)
        
        # Generate mock client results
        mock_results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.parameters = generate_mock_client_parameters(1, 100)[0]
            mock_result.num_examples = 50
            mock_results.append((mock_result, i))
        
        # Test aggregation
        aggregated_params, metrics = strategy.aggregate_fit(
            server_round=1,
            results=mock_results,
            failures=[]
        )
        
        assert aggregated_params is not None
        assert len(aggregated_params) > 0

    def test_aggregate_fit_with_removal(self):
        """Test client removal logic when round >= begin_removing_from_round."""
        strategy = YourStrategy(your_param=0.8, begin_removing_from_round=2)
        
        # Test with malicious clients (should be removed)
        mock_results = self._create_mixed_client_results()
        
        aggregated_params, metrics = strategy.aggregate_fit(
            server_round=3,  # >= begin_removing_from_round
            results=mock_results,
            failures=[]
        )
        
        # Verify removal occurred
        assert "removed_clients" in metrics
        assert len(metrics["removed_clients"]) > 0

    def test_edge_cases(self):
        """Test edge cases: empty results, single client, etc."""
        strategy = YourStrategy()
        
        # Test empty results
        aggregated_params, metrics = strategy.aggregate_fit(1, [], [])
        assert aggregated_params is None or len(aggregated_params) == 0
        
        # Test single client
        single_result = [(Mock(), 0)]
        single_result[0][0].parameters = generate_mock_client_parameters(1, 100)[0]
        single_result[0][0].num_examples = 50
        
        aggregated_params, metrics = strategy.aggregate_fit(1, single_result, [])
        assert aggregated_params is not None

    def _create_mixed_client_results(self):
        """Helper method to create honest and malicious client results."""
        # Implementation specific to your strategy's needs
        pass
```

**Step 3**: Add strategy to parameterized tests

```python
# Add to tests/unit/test_simulation_strategies/test_strategy_variations.py
STRATEGY_CONFIGS = {
    # ... existing strategies ...
    "your_strategy": {
        "aggregation_strategy_keyword": "your_strategy",
        "your_param": 0.7,
        "begin_removing_from_round": 2,
    }
}
```

### 2️⃣ New Data Model

When adding models to `src/data_models/`:

**Step 1**: Create test file

```bash
touch tests/unit/test_data_models/test_your_model.py
```

**Step 2**: Follow data model test pattern

```python
"""Unit tests for YourModel data class."""

import pytest
from src.data_models.your_model import YourModel


class TestYourModel:
    """Test suite for YourModel data validation and operations."""

    def test_initialization_valid_data(self):
        """Test model initialization with valid parameters."""
        model = YourModel(
            param1="valid_value",
            param2=42,
            param3=3.14
        )
        assert model.param1 == "valid_value"
        assert model.param2 == 42
        assert model.param3 == 3.14

    def test_initialization_invalid_data(self):
        """Test model validation with invalid parameters."""
        with pytest.raises(ValueError, match="param1 cannot be empty"):
            YourModel(param1="", param2=42, param3=3.14)
            
        with pytest.raises(ValueError, match="param2 must be positive"):
            YourModel(param1="valid", param2=-1, param3=3.14)

    def test_from_dict_method(self):
        """Test model creation from dictionary."""
        config_dict = {
            "param1": "test_value",
            "param2": 100,
            "param3": 2.71
        }
        model = YourModel.from_dict(config_dict)
        assert model.param1 == "test_value"
        assert model.param2 == 100

    def test_to_json_method(self):
        """Test model serialization to JSON."""
        model = YourModel("test", 50, 1.41)
        json_str = model.to_json()
        assert "test" in json_str
        assert "50" in json_str

    def test_required_fields_validation(self):
        """Test that all required fields are validated."""
        with pytest.raises(TypeError):
            YourModel()  # Missing required parameters
```

### 3️⃣ New Network Model

When adding to `src/network_models/`:

**Step 1**: Add test in `tests/unit/test_network_models/test_network_definitions.py`

```python
def test_your_network_initialization():
    """Test YourNetwork model initialization."""
    from src.network_models.your_network_definition import YourNetwork
    
    model = YourNetwork(num_classes=10, input_channels=3)
    assert model.num_classes == 10
    assert model.input_channels == 3

def test_your_network_forward_pass():
    """Test YourNetwork forward pass with mock data."""
    from src.network_models.your_network_definition import YourNetwork
    
    model = YourNetwork()
    mock_input = torch.randn(4, 3, 32, 32)  # batch_size=4
    
    output = model(mock_input)
    assert output.shape == (4, 10)  # Expected output shape

def test_your_network_get_parameters():
    """Test parameter extraction for federated learning."""
    from src.network_models.your_network_definition import YourNetwork
    
    model = YourNetwork()
    params = model.get_parameters()
    
    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)
```

## 🤖 Adding Tests for LLM-Generated Code

### 1️⃣ Validate LLM Output Structure

When using LLMs to generate new federated learning components:

**Step 1**: Create validation tests first

```python
def test_llm_generated_strategy_interface():
    """Validate LLM-generated strategy follows required interface."""
    from src.simulation_strategies.llm_generated_strategy import LLMGeneratedStrategy
    
    strategy = LLMGeneratedStrategy()
    
    # Verify required methods exist
    assert hasattr(strategy, 'aggregate_fit')
    assert hasattr(strategy, 'aggregate_evaluate') 
    assert callable(strategy.aggregate_fit)
    assert callable(strategy.aggregate_evaluate)
    
    # Verify method signatures
    import inspect
    fit_sig = inspect.signature(strategy.aggregate_fit)
    expected_params = ['server_round', 'results', 'failures']
    assert all(param in fit_sig.parameters for param in expected_params)
```

**Step 2**: Test mathematical correctness

```python
def test_llm_generated_algorithm_correctness():
    """Test mathematical correctness of LLM-generated algorithm."""
    strategy = LLMGeneratedStrategy(threshold=0.5)
    
    # Create known test case
    honest_params = [np.ones(10) * 1.0, np.ones(10) * 1.1, np.ones(10) * 0.9]
    malicious_params = [np.ones(10) * 10.0]  # Obvious outlier
    
    mock_results = []
    for params in honest_params + malicious_params:
        result = Mock()
        result.parameters = params
        result.num_examples = 50
        mock_results.append((result, len(mock_results)))
    
    aggregated, metrics = strategy.aggregate_fit(
        server_round=2,
        results=mock_results, 
        failures=[]
    )
    
    # Verify outlier was handled correctly
    # Expected behavior depends on your algorithm
    expected_range = (0.8, 1.2)  # Reasonable range for honest clients
    assert expected_range[0] <= np.mean(aggregated) <= expected_range[1]
```

### 2️⃣ Test Integration with Existing Framework

```python
def test_llm_code_integration():
    """Test LLM-generated code integrates properly with framework."""
    from src.simulation_strategies.llm_generated_strategy import LLMGeneratedStrategy
    from src.data_models.strategy_config import StrategyConfig
    
    # Test with framework configuration
    config = StrategyConfig.from_dict({
        "aggregation_strategy_keyword": "llm_generated",
        "num_of_rounds": 3,
        "num_of_clients": 10,
        "custom_param": 0.7
    })
    
    strategy = LLMGeneratedStrategy(custom_param=config.custom_param)
    
    # Should integrate without errors
    assert strategy.custom_param == 0.7
```

## ✨ Testing Best Practices

### 1️⃣ Test Organization

- **One test class per source class**: `TestYourStrategy` for `YourStrategy`
- **Descriptive test names**: `test_aggregate_fit_with_byzantine_clients()`
- **Group related tests**: Use test classes to organize functionality

### 2️⃣ Mock Data Guidelines

```python
# Use framework fixtures
from tests.fixtures.mock_datasets import (
    generate_mock_client_parameters,
    generate_byzantine_client_parameters,
    MockFederatedDataset
)

# Create realistic test scenarios
def test_with_realistic_data(self):
    """Test with federated learning specific data patterns."""
    # Generate heterogeneous client data
    federated_data = MockFederatedDataset(
        num_clients=10,
        samples_per_client=100,
        input_shape=(3, 32, 32)  # Matches real dataset
    )
    
    # Test your component with realistic FL data
    strategy = YourStrategy()
    result = strategy.process_federated_data(federated_data)
    
    assert result is not None
```

### 3️⃣ Parameter Validation Testing

```python
def test_parameter_validation(self):
    """Test all parameter edge cases and validation."""
    # Test boundary values
    strategy = YourStrategy(threshold=0.0)  # Minimum
    assert strategy.threshold == 0.0
    
    strategy = YourStrategy(threshold=1.0)  # Maximum
    assert strategy.threshold == 1.0
    
    # Test invalid values
    with pytest.raises(ValueError):
        YourStrategy(threshold=-0.1)  # Below minimum
        
    with pytest.raises(ValueError):
        YourStrategy(threshold=1.1)   # Above maximum
```

### 4️⃣ Byzantine Resistance Testing

```python
def test_byzantine_resistance(self):
    """Test strategy resistance against Byzantine attacks."""
    strategy = YourStrategy(byzantine_threshold=0.3)
    
    # Generate mixed client parameters
    honest_params = generate_mock_client_parameters(7, 1000)
    byzantine_params = generate_byzantine_client_parameters(
        num_clients=3,
        num_byzantine=3,
        attack_type="gaussian_noise"
    )
    
    all_results = []
    for i, params in enumerate(honest_params + byzantine_params):
        result = Mock()
        result.parameters = params
        result.num_examples = 50
        all_results.append((result, i))
    
    aggregated, metrics = strategy.aggregate_fit(1, all_results, [])
    
    # Verify Byzantine resilience
    assert "removed_clients" in metrics
    # Strategy should remove Byzantine clients
    assert len(metrics["removed_clients"]) >= 2
```

## 🏃 Running Your Tests

### 🎯 Execute specific test files

```bash
# Test your new strategy
pytest tests/unit/test_simulation_strategies/test_your_strategy.py -v

# Test your new data model
pytest tests/unit/test_data_models/test_your_model.py -v

# Test integration with existing framework
pytest tests/integration/ -k "your_component" -v
```

### 🔍 Analyze test suite output

```bash
# Generate detailed test output for analysis
python -m pytest -v --tb=short -s > pytest.log 2>&1

# View the log file to analyze test results
cat pytest.log

# Run specific test with detailed output
python -m pytest tests/unit/test_simulation_strategies/test_your_strategy.py -v --tb=short -s
```

### 📊 Check test coverage

```bash
# Generate coverage report for your new code
pytest --cov=src.simulation_strategies.your_strategy --cov-report=html tests/unit/test_simulation_strategies/test_your_strategy.py
```

### 📊 Run performance tests

```bash
# Test scalability of your new component
pytest tests/performance/test_scalability.py::test_strategy_scalability[your_strategy] -v
```

### ✨ Code quality checks

```bash
# Check code formatting and style
flake8 --ignore=E501,W503,E203 tests

# Sort imports properly
isort tests

# Check specific test file
flake8 --ignore=E501,W503,E203 tests/unit/test_simulation_strategies/test_your_strategy.py
isort tests/unit/test_simulation_strategies/test_your_strategy.py
```

## 🚨 Common Beginner Mistakes

### 1️⃣ Wrong Directory

```bash
# ❌ WRONG - Running from tests/ directory
cd tests/
python -m pytest unit/test_my_first_test.py  # Will fail with import errors

# ✅ RIGHT - Always run from project root
cd /path/to/fl-execution-framework
python -m pytest tests/unit/test_my_first_test.py
```

### 2️⃣ Coverage Blocking Your Progress

```bash
# ❌ This will fail for beginners (coverage too low)
python -m pytest tests/unit/test_my_first_test.py

# ✅ Use this instead while learning
python -m pytest tests/unit/test_my_first_test.py --no-cov -v -s
```

### 3️⃣ Python Command Issues

```bash
# Check which Python you're using
python --version   # Should show Python 3.10+
python3 --version  # Alternative on some systems

# If "python" shows Python 2.x or doesn't exist, use "python3" instead:
python3 -m pytest tests/unit/test_data_models/ --no-cov -v

# Check if you're in the right virtual environment:
pip list | grep pytest  # Should show pytest installed

# If not, activate your environment:
source .venv/Scripts/activate  # Windows Git Bash
source .venv/bin/activate      # Linux/Mac
```

### 4️⃣ Test File Naming

```bash
# ❌ WRONG - pytest won't find these
my_test.py
testing_stuff.py

# ✅ RIGHT - MUST start with "test_"
test_my_first_test.py
test_learning_basics.py
```

## ⚠️ Common Testing Pitfalls

### 1️⃣ Insufficient Edge Case Coverage

❌ **Wrong**: Only testing happy path

```python
def test_strategy(self):
    strategy = YourStrategy()
    result = strategy.aggregate_fit(1, normal_results, [])
    assert result is not None
```

✅ **Right**: Test edge cases

```python
def test_strategy_edge_cases(self):
    strategy = YourStrategy()
    
    # Test empty results
    assert strategy.aggregate_fit(1, [], []) is None
    
    # Test single client
    single_result = [create_mock_result()]
    result = strategy.aggregate_fit(1, single_result, [])
    assert result is not None
    
    # Test all malicious clients
    malicious_results = create_all_malicious_results()
    result = strategy.aggregate_fit(1, malicious_results, [])
    # Should handle gracefully
```

### 2️⃣ Not Testing Framework Integration

❌ **Wrong**: Testing in isolation only
✅ **Right**: Add integration tests in `tests/integration/test_strategy_combinations.py`

### 3️⃣ Forgetting Performance Impact

❌ **Wrong**: No performance considerations
✅ **Right**: Add performance test in `tests/performance/test_scalability.py`

## 🔧 Troubleshooting Common Issues

### 🔍 Test Discovery Problems

```bash
# If pytest can't find your tests:
python -m pytest tests/unit/your_test.py -v

# If imports fail:
# Make sure you're in the project root directory
# Check that src/ has __init__.py files
```

### 🎭 Mock Data Issues

```bash
# If mock data seems wrong:
python -c "from tests.fixtures.mock_datasets import *; print('Fixtures loaded!')"

# If shapes don't match:
# Check input_shapes in mock_datasets.py for your dataset type
```

### ❌ Common Test Failures

- **Import Errors**: Ensure you're using `from src.module` imports
- **Shape Mismatches**: Verify mock data matches your expected input shapes
- **Assertion Failures**: Add print statements to debug: `python -m pytest -s`

## 📞 Getting Help

### 📚 Quick References

- **Mock Data**: Use existing fixtures in `tests/fixtures/`
- **Test Patterns**: Copy patterns from similar existing tests
- **Integration**: Check `tests/integration/` for multi-component testing examples
- **Performance**: Reference `tests/performance/` for scalability testing patterns

### 🎓 Learning Resources

- **Start Here**: Run `python -m pytest tests/unit/test_data_models/ -v` to see simple examples
- **Advanced Patterns**: Study `tests/unit/test_simulation_strategies/` for complex scenarios
- **Mock Strategies**: Explore `tests/fixtures/mock_datasets.py` for data generation

### 💡 Workflow Tips

1. **Start simple** - Get one test passing before adding complexity
2. **Use existing patterns** - Copy and modify similar tests
3. **Test incrementally** - Run tests frequently during development
4. **Ask for help** - The testing framework is designed to be approachable

---

## 📋 Checklist for New Researchers

### ✅ Before You Start

- [ ] Environment activated and dependencies installed
- [ ] Test suite setup validated: `python tests/validate_coverage_setup.py`
- [ ] Read `tests/docs/README.md` for architecture overview
- [ ] Run existing tests to verify setup: `python -m pytest tests/unit/test_data_models/ -v`

### 🎆 Adding Your First Test

- [ ] Created test file following naming convention (`test_*.py`)
- [ ] Used existing fixtures for mock data
- [ ] Followed AAA pattern (Arrange, Act, Assert)
- [ ] Added docstrings to test functions

### 📤 Before Submitting

- [ ] All new tests pass: `python -m pytest tests/unit/your_test_file.py -v`
- [ ] Code style check: `flake8 --ignore=E501,W503,E203 tests/unit/your_test_file.py`
- [ ] Import sorting: `isort tests/unit/your_test_file.py`
- [ ] Edge cases covered (empty inputs, invalid data, etc.)

This systematic approach maintains the framework's quality standards while providing validation for federated learning scenarios.
