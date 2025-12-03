# Artifact Description: FL Execution Framework

## Overview

Federated Learning simulation framework for evaluating Byzantine-resilient aggregation strategies under poisoning attacks.

## Claims Checklist

| Paper Claim | Artifact Location | Verification Command |
|-------------|-------------------|---------------------|
| Krum defense effectiveness | `config/simulation_strategies/testing/femnist_krum_*.json` | `pytest tests/integration/test_strategy_pipeline.py -k krum` |
| Multi-Krum aggregation | `config/simulation_strategies/testing/femnist_mkrum_*.json` | `pytest tests/integration/test_strategy_pipeline.py -k multi_krum` |
| Bulyan strategy | `config/simulation_strategies/testing/femnist_bulyan_*.json` | `pytest tests/integration/test_strategy_pipeline.py -k bulyan` |
| RFA defense | `config/simulation_strategies/testing/femnist_rfa_*.json` | `pytest tests/integration/test_strategy_pipeline.py -k rfa` |
| Trust-based removal | `config/simulation_strategies/testing/femnist_trust_*.json` | `pytest tests/integration/test_strategy_pipeline.py -k trust` |
| PID-based defense | `config/simulation_strategies/testing/femnist_pid*.json` | `pytest tests/integration/test_strategy_pipeline.py -k pid` |
| Trimmed Mean | `config/simulation_strategies/testing/femnist_trimmean_*.json` | `pytest tests/integration/test_strategy_pipeline.py -k trimmed` |

## Hardware Requirements

- **CPU**: Any modern x86_64 processor
- **GPU**: Optional (CUDA 11.8+ for GPU-accelerated training)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for dependencies + dataset space

## Software Dependencies

- Python 3.9, 3.10, or 3.11
- See `requirements.txt` for full dependency list
- See `src/api/requirements.txt` for API dependencies

## Installation

```bash
# Clone repository
git clone <repository-url>
cd fl-execution-framework

# Install dependencies
pip install -r requirements.txt

# Verify installation
PYTHONPATH=. python tests/scripts/ci_smoke_test.py --parse-only
```

## Smoke Test

```bash
# Config validation only (fast, no GPU needed)
PYTHONPATH=. python tests/scripts/ci_smoke_test.py --parse-only

# Run strategy pipeline integration tests
pytest tests/integration/test_strategy_pipeline.py -v

# Run full test suite
pytest tests/ -v
```

## Expected Results

Baseline fixtures in `tests/fixtures/baselines/` define expected simulation outputs.
These baselines were recorded using `tests/scripts/record_baselines.py` and serve as regression tests.

## Experiment Reproduction

To reproduce paper experiments:

1. Select appropriate config from `config/simulation_strategies/testing/`
2. Run: `python src/simulation_runner.py <config_path>`
3. Results saved to `out/` directory with plots and CSV exports

## Directory Structure

```
fl-execution-framework/
├── config/simulation_strategies/testing/   # Test configurations
├── tests/
│   ├── fixtures/baselines/                 # Expected output baselines
│   ├── integration/                        # Integration tests
│   ├── scripts/                            # CI smoke test scripts
│   └── unit/                               # Unit tests
├── src/                                    # Source code
├── docs/                                   # Documentation
└── requirements.txt                        # Dependencies
```