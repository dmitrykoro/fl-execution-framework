# FL Framework Testing & Development

**Technical guide for FL execution framework testing and development.**

---

## ⚡ Quick Start Commands

```bash
# Essential validation
python tests/validate_coverage_setup.py
python -m pytest --version
PYTHONPATH=. python tests/demo/strategy_config_demo.py

# Development workflow
cd tests && ./lint.sh                                    # Quality check + type hints
python -m pytest tests/unit/ -n auto -x --tb=line       # Fast parallel unit tests
python -m pytest tests/unit/test_module.py::test_name -v # Single test
```

### 🚀 Advanced Commands

```bash
# Comprehensive testing
cd tests && ./lint.sh --test    # Adds pytest, all checks
cd tests && ./lint.sh --sonar   # Adds code quality analysis

# Parallel execution (recommended)
pytest -n auto tests/unit/ -v          # Unit tests in parallel
pytest -n 0 tests/integration/ -v      # Integration tests serial (required)
pytest -n 0 tests/performance/ -v       # Performance benchmarks (serial)
```

---

## 🔧 Strategy-Based Client Configuration System

**Issue Addressed:** Researchers were using Flower's production defaults (`min_fit_clients = 2`) causing convergence failures in controlled research environments where Byzantine-robust strategies need ALL clients.

**✅ Auto-Configuration:**

- **Byzantine strategies** (`trust`, `krum`, `multi-krum`, `pid`) → `min_fit_clients = num_of_clients`
- **Flexible strategies** (`trimmed_mean`) → 80% participation with warnings
- **Unknown strategies** → Conservative 60% with research recommendations

**📚 Demo & Validation:**

```bash
PYTHONPATH=. python tests/demo/strategy_config_demo.py
```

**Key Features:**

- Strategy-aware auto-configuration preventing convergence failures
- Research-mode optimization vs production defaults
- Validation warnings for configuration issues
- Manual override support with intelligent warnings

---

## 🧠 Federated Learning Fundamentals

### 🎪 How FL Works

1. **Initialization**: Server creates global model
2. **Distribution**: Model sent to selected clients
3. **Local Training**: Clients train on private data
4. **Update Collection**: Parameter updates sent to server
5. **Aggregation**: Server combines updates using strategy
6. **Model Update**: New global model created
7. **Repeat**: Continue until convergence

### 🛡️ Byzantine-Robust Strategies

**Trust-Based (`trust`):**

- Builds client reputation over rounds
- Requires consistent participation for reputation scores
- **Critical**: `min_fit_clients = num_of_clients` for proper operation

**Krum Family (`krum`, `multi-krum`):**

- Distance-based Byzantine detection
- Needs all clients for accurate distance calculations
- **Critical**: Full participation required

**PID Controllers (`pid`, `pid_scaled`, `pid_standardized`):**

- Feedback control for adaptive aggregation
- Requires consistent client participation for stable convergence
- **Critical**: Disrupted by variable participation

### 🔄 Flexible Strategies

**Trimmed Mean (`trimmed_mean`):**

- Removes outliers before averaging
- Can work with variable participation
- **Optimal**: Better results with consistent participation

**Bulyan (`bulyan`):**

- Two-stage: Multi-Krum filtering + Trimmed-Mean aggregation
- Requires specific client ratios for Byzantine robustness
- **Critical**: Needs consistent participation

---

## 🧪 Test Development Standards

### 🐍 Python Standards (NON-NEGOTIABLE)

**Required:**

- Type hints: `def func(x: int) -> str:`
- Google-style docstrings for public functions
- Specific exceptions, not bare `except:`
- Absolute imports: `from src.module`

**Test Standards:**

- AAA Pattern: Arrange, Act, Assert
- Descriptive names: `test_should_return_error_when_input_invalid`
- One concept per test
- Mock external dependencies

### 🗂️ Test Organization

**Directory Structure:**

```text
tests/
├── unit/                     # Fast isolated tests
│   ├── test_data_models/    # Model validation
│   ├── test_aggregation/    # Strategy testing
│   └── test_config_loaders/ # Configuration logic
├── integration/             # Component interaction (serial execution)
├── performance/             # Scalability benchmarks
├── demo/                    # Runnable examples
└── docs/                    # This guide
```

**Execution Strategy:**

- **Unit tests**: Parallel execution (`-n auto`) for speed
- **Integration tests**: Serial execution (`-n 0`) for isolation
- **Performance tests**: Single worker for accurate benchmarking

### 🎲 Test Data Generation

**Mock Client Data:**

```python
from tests.conftest import generate_mock_client_data

client_results = generate_mock_client_data(
    num_clients=5,
    param_shape=(10, 5)  # Configurable tensor dimensions
)
# Returns: List[Tuple[ClientProxy, FitRes]]
```

**Dataset Type Support:**

- `its`, `flair`, `lung_photos`: (3, 224, 224) RGB high-res
- `femnist_iid/niid`, `pneumoniamnist`: (1, 28, 28) grayscale
- `bloodmnist`: (3, 28, 28) RGB medical
- `mock`: (3, 32, 32) general testing

**Attack Simulation:**

```python
def generate_byzantine_updates(num_malicious: int, attack_type: str):
    """Generate Byzantine attack patterns for robustness testing."""
```

### 📝 Test Patterns

**1. Strategy Testing:**

```python
def test_trust_strategy_with_full_participation():
    """Test trust strategy with research-appropriate client participation."""
    config = {"aggregation_strategy_keyword": "trust", "num_of_clients": 5}
    config = apply_smart_client_config(config)
    assert config["min_fit_clients"] == 5  # Auto-configured correctly
```

**2. Configuration Validation:**

```python
def test_config_validation():
    """Test strategy config prevents convergence issues."""
    config = {"aggregation_strategy_keyword": "krum", "num_of_clients": 8}
    issues = validate_client_config(config)
    assert any("CONVERGENCE RISK" in issue for issue in issues)
```

**3. Mock Data Testing:**

```python
def test_aggregation_with_mock_data():
    """Test aggregation strategies with synthetic FL data."""
    client_results = generate_mock_client_data(num_clients=10)
    strategy = TrustStrategy()
    aggregated = strategy.aggregate(client_results)
    assert aggregated.parameters.shape == expected_shape
```

---

## 🛠️ Development Workflow

```bash
# 1. Fast feedback loop
python -m pytest tests/unit/test_module.py::TestClass::test_method -v

# 2. Quick validation
python -m pytest tests/unit/ -n auto -x --tb=line

# 3. Quality check
cd tests && ./lint.sh

# 4. Full validation (session end)
cd tests && ./lint.sh --test
```

### 🚨 Session End Checklist

- [ ] Tests pass: `python -m pytest tests/`
- [ ] Code quality: `cd tests && ./lint.sh`
- [ ] Changes primarily in `tests/` directory
- [ ] Any `src/` changes documented in `refactoring_for_testability.md`

---

## 📊 Performance & Scalability

### 🚀 Parallel Execution Benefits

**Performance:**

- ⚡ Unit tests: ~50% faster with parallel workers
- 🔧 CI pipeline: Faster feedback cycles
- 🛠️ Local development: Auto-scaling based on CPU cores
- 📊 Scalability: Better resource utilization

**Execution Guidelines:**

- ✅ **Unit tests**: Safe for parallel (`-n auto`)
- ❌ **Performance tests**: Must run serial (`-n 0`) for timing accuracy
- ❌ **Integration tests**: Must run serial (`-n 0`) to prevent conflicts

### 📈 Scalability Testing

**Client Scaling Tests:**

```python
@pytest.mark.parametrize("num_clients", [10, 25, 50, 100])
def test_strategy_scales_linearly(num_clients):
    """Verify O(n) complexity for client scaling."""
```

**Parameter Size Tests:**

```python
@pytest.mark.parametrize("param_size", [1000, 10000, 100000])
def test_aggregation_scales_with_parameters(param_size):
    """Verify aggregation handles large parameter spaces."""
```

---

## 🔍 Quality Assurance

### 📊 Coverage Standards

```bash
# Generate coverage reports
pytest --cov=src --cov-report=html --cov-report=term
pytest --cov=src --cov-fail-under=70  # Minimum 70% coverage
```

### 🛡️ Code Quality Tools

**Integrated in `tests/lint.sh`:**

- **ruff**: Fast linting and formatting
- **mypy**: Static type checking
- **pyright**: Additional type validation
- **pytest**: Comprehensive test execution

**SonarQube (Optional):**

```bash
cd tests && ./lint.sh --sonar  # Advanced code quality analysis
```

### 🚨 Common Issues & Solutions

**Import Errors:**

```bash
# Use PYTHONPATH for demos
PYTHONPATH=. python tests/demo/strategy_config_demo.py
```

**Type Errors:**

- All functions must have type hints
- Use `Optional[T]` for nullable parameters
- Import typing modules: `from typing import List, Dict, Optional`

**Test Failures:**

- Check `pytest.log` for detailed failure analysis
- Use `-v -s` flags for verbose output
- Single test debugging: `pytest path/to/test::test_name -v`

---

## 🎯 Specialized Testing Areas

### 🔧 Configuration Testing

**Strategy-Based Client Config:**

```python
def test_byzantine_strategy_auto_config():
    """Test Byzantine strategies get proper client participation."""
    config = {"aggregation_strategy_keyword": "trust", "num_of_clients": 6}
    updated = apply_smart_client_config(config)
    assert updated["min_fit_clients"] == 6
    assert updated["min_evaluate_clients"] == 6
```

**Validation Testing:**

```python
def test_convergence_warnings():
    """Test system warns about convergence risks."""
    config = {"aggregation_strategy_keyword": "krum", "num_of_clients": 10, "min_fit_clients": 5}
    issues = validate_client_config(config)
    assert any("CONVERGENCE RISK" in issue for issue in issues)
```

### 🤖 Strategy Testing Patterns

**Trust Strategy:**

```python
def test_trust_builds_reputation():
    """Test trust strategy builds client reputation over rounds."""
    # Simulate multiple rounds with consistent client participation
    # Verify reputation scores evolve correctly
```

**Krum Strategy:**

```python
def test_krum_detects_byzantine():
    """Test Krum correctly identifies Byzantine clients."""
    # Generate normal + Byzantine client updates
    # Verify Byzantine clients are excluded from aggregation
```

**PID Strategy:**

```python
def test_pid_convergence():
    """Test PID controller achieves stable convergence."""
    # Simulate feedback loops with consistent participation
    # Verify convergence metrics improve over rounds
```

### 🎭 Attack Simulation

**Byzantine Attacks:**

```python
def generate_byzantine_attack(attack_type: str, intensity: float):
    """Generate Byzantine attack patterns for robustness testing.

    Args:
        attack_type: 'random', 'sign_flip', 'gaussian_noise'
        intensity: Attack strength multiplier
    """
```

**Defense Validation:**

```python
def test_defense_against_attack():
    """Test aggregation strategy defends against Byzantine attacks."""
    # Mix normal and malicious client updates
    # Verify final model maintains accuracy despite attacks
```

---

## 🏆 Success Patterns

**Excellent Session:**

- Zero `src/` changes, creative test-only solutions
- All tests pass, high coverage maintained
- Clear, focused changes with proper documentation

**Good Session:**

- Minimal, well-documented `src/` bug fixes
- Proper risk assessment for any production changes
- Maintains framework integrity

**Improvement Needed:**

- Multiple undocumented `src/` changes
- Creating files unnecessarily
- Breaking existing functionality

---

## 📚 Additional Resources

**Demo Scripts:**

- `tests/demo/strategy_config_demo.py` - Strategy-based config in action
- `tests/demo/failure_logging_demo.py` - Test failure analysis
- `tests/demo/mock_data_showcase.py` - FL data generation patterns

**Historical Documentation:**

- `tests/docs/refactoring_for_testability.md` - Production changes log
- Framework change history with risk assessments
- Guidelines for minimal, necessary `src/` modifications

**Quick Reference:**

- All commands assume project root directory
- Use `cd tests && ./lint.sh` for quality checks
- Parallel execution: `pytest -n auto tests/unit/`
- Serial execution: `pytest -n 0 tests/integration/`
