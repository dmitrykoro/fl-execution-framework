# CI Smoke Tests

## Quick Start

```bash
# Run smoke tests (output validation)
PYTHONPATH=. python tests/scripts/ci_smoke_test.py

# Run strategy integration tests
python -m pytest tests/integration/test_strategy_pipeline.py -v

# Record a new baseline
PYTHONPATH=. python tests/scripts/record_baselines.py --config your_config.json
```

## Files

- `tests/scripts/ci_smoke_test.py` - Validates output generation with mock data
- `tests/integration/test_strategy_pipeline.py` - Runs actual strategy code with mock weights
- `tests/fixtures/baselines/` - Pre-recorded metrics (24 configs)
- `.github/workflows/ci-smoke-tests.yml` - CI workflow (runs both tests in parallel)
