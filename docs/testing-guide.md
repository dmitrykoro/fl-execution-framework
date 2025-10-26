# 🧪 FL Testing Guide

**Technical guide for testing federated learning strategies.**

---

## ⚡ Essential Commands

```bash
# Daily workflow
cd tests && ./lint.sh                     # Quality check + fixes
python -m pytest tests/unit/              # Unit tests
python -m pytest tests/unit/test_file.py::test_name -v  # Single test

# Full validation
python -m pytest tests/                   # All tests
python tests/demo/mock_data_showcase.py  # See mock data generation
```

---

## 🎯 Adding New Tests

### 1. Pick the Right Directory

```text
tests/
├── unit/                    # ← Add most tests here (fast, parallel)
│   ├── test_data_models/   # Model/config testing
│   └── test_strategies/    # Strategy algorithm testing
├── integration/            # Component interaction (slower, serial)
└── demo/                   # Runnable examples
```

### 2. Use the Test Template

```python
"""Test module for [YourComponent]."""

import pytest
from src.your_module import YourClass
from tests.conftest import generate_mock_client_data


class TestYourClass:
    """YourClass unit tests."""

    @pytest.fixture
    def your_instance(self):
        """Create test instance."""
        return YourClass(param1=value1, param2=value2)

    def test_should_do_something_when_condition(self, your_instance):
        """Test descriptive behavior description."""
        # Arrange
        input_data = generate_mock_client_data(num_clients=3)

        # Act
        result = your_instance.process(input_data)

        # Assert
        assert result is not None
        assert len(result) == 3
```

### 3. Test Strategy Algorithms

```python
def test_trust_strategy_builds_reputation(self):
    """Test trust strategy increases scores for consistent clients."""
    # Use mock clients with consistent behavior
    client_results = generate_mock_client_data(num_clients=5)

    strategy = TrustStrategy()
    strategy.aggregate_fit(1, client_results, [])

    # Verify reputation scores are calculated
    assert len(strategy.client_scores) == 5
```

---

## 🔧 Strategy Auto-Configuration

**Problem**: Users use production defaults → convergence failures
**Solution**: Auto-configure based on strategy type

```python
# Byzantine strategies need ALL clients
config = {"aggregation_strategy_keyword": "trust", "num_of_clients": 5}
config = apply_smart_client_config(config)
assert config["min_fit_clients"] == 5  # Auto-set to prevent failures

# See unit tests for config examples:
python -m pytest tests/unit/test_config_loaders/ -v
```

**Key Strategy Types:**

- **Byzantine** (`trust`, `krum`, `multi-krum`, `pid`) → 100% participation required
- **Flexible** (`trimmed_mean`) → 80% participation recommended
- **Unknown** → 60% participation with warnings

---

## 🎲 Mock Data Generation

```python
from tests.conftest import generate_mock_client_data

# Basic usage
client_results = generate_mock_client_data(num_clients=6)

# Custom parameters
client_results = generate_mock_client_data(
    num_clients=10,
    param_shape=(5, 5)  # Custom tensor dimensions
)

# Returns: List[Tuple[ClientProxy, FitRes]] ready for strategy testing
```

---

## 🏗️ Code Standards

### Required

- **Type hints**: `def process(data: List[int]) -> Dict[str, float]:`
- **Descriptive names**: `test_should_handle_empty_client_list`
- **AAA pattern**: Arrange, Act, Assert
- **Mock externals**: Use fixtures for dependencies

### Test File Structure

```python
"""Brief module description."""

import pytest
from src.module import Class
from tests.conftest import helpers


class TestClass:
    """Class description."""

    @pytest.fixture
    def setup_data(self):
        """Fixture description."""
        return test_data

    def test_behavior_description(self, setup_data):
        """Test what happens when conditions are met."""
        # Arrange, Act, Assert
```

---

## 🚀 Performance Tips

**Parallel Execution (pytest-xdist):**

```bash
# Parallel execution for speed
pytest -n auto tests/unit/        # Auto-detect CPU cores, faster unit tests
pytest -n 4 tests/unit/           # Use 4 workers explicitly

# Serial execution when needed
pytest -n 0 tests/integration/    # Force serial, prevents conflicts
pytest -n 0 tests/performance/    # Serial for accurate timing
```

**Quick Feedback:**

```bash
pytest tests/unit/test_file.py::TestClass::test_method -v  # Single test
pytest tests/unit/ -x --tb=line   # Stop on first failure, minimal output
```

**Why Use These Flags:**

- `-n auto` = Use all CPU cores, ~50% faster for unit tests
- `-n 0` = Force serial execution, required for integration/performance tests
- `-n 4` = Use specific number of workers

**Why Serial Execution Required:**

- **Integration tests**: Share resources (files, networks), parallel = conflicts
- **Performance tests**: Need accurate timing, parallel = skewed results
- **Unit tests**: Isolated by design, safe for parallel execution

---

## 🔍 Common Issues

**Import errors**:

```bash
python tests/demo/script.py  # For demos
```

**Type errors**: All functions need type hints

```python
# Bad
def process(data):
    return data

# Good
def process(data: List[str]) -> List[str]:
    return data
```

**Test failures**: Use verbose mode for details

```bash
pytest tests/unit/test_file.py -v -s  # Verbose with print statements
```

---

## 📊 Quality Checks

```bash
cd tests && ./lint.sh              # Format, lint, type check
cd tests && ./lint.sh --test       # Add full test run
cd tests && ./lint.sh --sonar      # Advanced quality analysis
```

> **Note on SonarQube:**
> After running the analysis, view the report at `http://localhost:9000`. The first time you access it, log in with the default credentials:
>
> - **Username:** `admin`
> - **Password:** `admin`
>
> You will be prompted to change the password immediately.

```bash
# Coverage
pytest --cov=src --cov-report=html tests/
```

---

## 🎯 When to Add Tests

### Always Test

- New strategy algorithms
- Configuration validation logic
- Data processing functions
- Bug fixes

### Test Patterns

```python
# Strategy testing
def test_krum_selects_best_clients():
    """Test Krum aggregation selects expected clients."""

# Config testing
def test_config_warns_about_convergence_risk():
    """Test system warns when config may cause convergence issues."""

# Edge cases
def test_handles_empty_client_list():
    """Test strategy gracefully handles no participating clients."""
```

---

## 📚 Quick Reference

**Essential files**:

- `tests/conftest.py` - Mock data generators
- `tests/demo/` - Interactive examples and showcases
- `tests/docs/refactoring_for_testability.md` - Production change log

**Need help?** Check the demo scripts or existing tests for patterns!

---

**Remember**: Keep tests focused, use descriptive names, and leverage the mock data generators! 🎯
