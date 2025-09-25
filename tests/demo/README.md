# FL Framework Demos

Interactive examples demonstrating framework capabilities.

## Quick Commands

```bash
# Key demos (copy & paste these)
PYTHONPATH=. python tests/demo/smart_client_config_example.py
PYTHONPATH=. python tests/demo/mock_data_showcase.py
PYTHONPATH=. python -m pytest tests/demo/failure_logging_demo.py -v -s

# Run all as tests
PYTHONPATH=. python -m pytest tests/demo/ -v -s
```

## Demo Scripts

- **`strategy_client_config_example.py`** - Smart FL strategy auto-configuration
- **`mock_data_showcase.py`** - FL synthetic data generation (20x faster than real datasets)
- **`failure_logging_demo.py`** - Intelligent test failure analysis and debugging

## Complete Documentation

**📚 See `tests/docs/TESTING_GUIDE.md` for everything:**

- FL fundamentals & Byzantine-robust strategies
- Strategy client configuration system
- Test development standards & patterns
- Performance optimization & parallel execution
