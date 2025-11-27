# CI Smoke Test System

## Problem

FL simulations take 1-20+ minutes each, making comprehensive CI testing impractical for every push/PR.

## Solution

A mock simulation system that uses pre-recorded baseline data to run the full output generation pipeline (plots, CSVs, HTML reports) without actual Flower training.

## How It Works

1. **Record baselines once** - Run actual simulations and capture per-round/per-client metrics
2. **Mock simulations in CI** - Use baseline data to drive the output generation pipeline
3. **Verify outputs** - Check that all expected files (PDFs, CSVs, HTML) are generated correctly

This catches:
- Config syntax errors and missing fields
- Output generation bugs (plotting, CSV writing, HTML dashboards)
- Integration issues between components

## Files

```
tests/
├── scripts/
│   ├── ci_smoke_test.py        # CI validation script
│   ├── mock_simulation_runner.py  # Runs output pipeline with mock data
│   └── record_baselines.py     # Records baselines from real simulations
└── fixtures/
    └── baselines/              # 24 recorded baselines (~15-25KB each)

.github/workflows/
└── ci-smoke-tests.yml          # GitHub Actions workflow
```

## CI Workflow

Triggers on push/PR when `config/`, `src/`, or `tests/scripts/` change:

```yaml
- run: python tests/scripts/ci_smoke_test.py
```

**Default mode**: Runs mock simulations for all 24 configs, verifies output generation.

**Parse-only mode** (`--parse-only`): Fast validation of config syntax only.

## Config Coverage

24 configs covering:
- **8 aggregation strategies**: Krum, Multi-Krum, Bulyan, PID (4 variants), RFA, Trust, Trimmed Mean
- **Attack types**: Label flip (20%, 50%), Gaussian noise, multi-attack combinations
- **Datasets**: FEMNIST, BreastMNIST

## Usage

```bash
# Run mock simulations (default: all 24 configs)
python tests/scripts/ci_smoke_test.py

# Test single config
python tests/scripts/ci_smoke_test.py --config femnist_krum_baseline.json

# Parse-only mode (faster, just validates config syntax)
python tests/scripts/ci_smoke_test.py --parse-only

# Record new baseline (after adding a config)
python tests/scripts/record_baselines.py --config my_new_config.json

# Re-record all baselines
python tests/scripts/record_baselines.py --all-fast
```

## Baseline Format

```json
{
  "config": "femnist_krum_baseline.json",
  "recorded_at": "2025-11-26T00:22:00",
  "framework_version": "1.0.0",
  "success": true,
  "num_clients": 10,
  "strategies": [
    {
      "total_rounds": 10,
      "final_accuracy": 75.9,
      "final_loss": 0.0349,
      "per_round": {
        "aggregated_loss": [0.085, 0.079, ...],
        "average_accuracy": [21.95, 42.32, ...]
      },
      "per_client": {
        "0": {
          "loss": [0.089, 0.078, ...],
          "accuracy": [0.21, 0.50, ...],
          "removal_criterion": [4.33, 5.34, ...],
          "absolute_distance": [0.60, 0.82, ...],
          "participation": [1, 1, 1, ...]
        }
      }
    }
  ],
  "outputs": {
    "plots": ["accuracy_history_0.pdf", ...],
    "csv": ["round_metrics_0.csv", ...],
    "attack_snapshots": 0
  },
  "runtime_seconds": 61.0
}
```

## Adding New Configs

1. Create config in `config/simulation_strategies/testing/`
2. Add filename to `FAST_CONFIGS` list in all three scripts
3. Run `python tests/scripts/record_baselines.py --config your_config.json`
4. Commit the new baseline JSON

## Generated Outputs

Each mock simulation generates:

| Type | Files |
|------|-------|
| PDFs | `accuracy_history_0.pdf`, `loss_history_0.pdf`, `removal_criterion_history_0.pdf`, `absolute_distance_history_0.pdf` |
| CSVs | `round_metrics_0.csv`, `per_client_metrics_0.csv`, `exec_stats_0.csv` |
| HTML | `index.html` (main dashboard) |
| JSON | `strategy_config_0.json` |
