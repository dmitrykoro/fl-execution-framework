# 🎭 FL Framework Demos

**Learn federated learning testing through interactive examples.**

## ⚡ Quick Start

```bash
# 1. Setup (required)
./reinstall_requirements.sh

# 2. Start here (recommended)
python -m tests.demo.run_showcase
```

## 📚 Learning Path

**New to FL testing?** Follow this order:

1. **`run_showcase`** - Safe cross-platform demo launcher
2. **`failure_logging_demo`** - Learn debugging skills
3. **`mock_data_showcase`** - Explore FL data generation
4. **Full suite** - Validate your understanding

## 🚀 Commands

```bash
# Recommended (cross-platform safe)
python -m tests.demo.run_showcase

# Direct execution (fastest)
python -m tests.demo.mock_data_showcase

# Debugging skills demo
python -m pytest tests/demo/failure_logging_demo.py -v -s

# Full validation suite
python -m pytest tests/demo/ -v -s
```

## 📋 Demo Scripts

### `mock_data_showcase.py` - FL Mock Data Generation

Demonstrates mock data for FL testing:

**✅ Working Examples:**

- Basic client generation (5 clients, FL structure)
- Heterogeneous data analysis (parameter variance > 1000)
- Byzantine attack simulation (3.6x higher magnitudes)
- Performance testing (50M+ parameters in <1s)
- Edge cases (empty, single, identical, large, tiny)

**❌ Expected Failures:**

- Strategy compatibility: `object of type 'Parameters' has no len()`
- Intentional demonstration of real integration challenges

**Features:**

- 20-30x faster than real data loading
- Deterministic (seed=42)
- Memory efficient
- Configurable parameter shapes

### `run_showcase.py` - Cross-Platform Demo Launcher

**Purpose**: Unicode-safe execution wrapper for consistent demo experience.

**Features:**

- Windows MINGW64 compatibility
- Platform detection and logging
- Safe demo execution across operating systems

### `failure_logging_demo.py` - Test Failure Analysis

**Purpose**: Learn systematic debugging for FL testing failures.

**Features:**

- Automatic failure detection/categorization
- Context-aware debugging hints
- Structured logging to `tests/logs/`
- Error heuristics (ImportError, FileNotFoundError, PyTorch)
- Interactive failure examples

**Usage:**

- Pytest: `python -m pytest tests/demo/failure_logging_demo.py -v -s`
- Uncomment methods in the code to trigger specific failures

## 📊 Expected Output

**Successful run shows:**

```bash
✅ Generated 5 clients
✅ Parameter variance > 1000 (heterogeneous data)
✅ Performance test: 50M+ parameters in <1s
```

**Symbol Guide:**

- **✅ Success** - Working correctly
- **❌ Educational failure** - Real-world FL challenges (intentional)
- **⚠️ Edge case** - Configuration issues or boundaries

## 🔧 Troubleshooting

**Import errors?** Run `./reinstall_requirements.sh` from root directory
**Unicode issues?** Use `python -m tests.demo.run_showcase`
**Windows issues?** Ensure MINGW64 bash environment
