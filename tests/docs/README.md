# FL Framework Testing Documentation

## 📚 Complete Developer Guide

**For testing and development documentation, see:**

### 🎯 [`TESTING_GUIDE.md`](TESTING_GUIDE.md)

**Everything you need in one place:**

- ⚡ Quick start commands & development workflow
- 🔧 Strategy client configuration system
- 🧠 FL fundamentals & Byzantine defense strategies
- 🧪 Test development standards & patterns
- 🚀 Performance optimization & parallel execution
- 📊 Quality assurance & scalability testing

## ⚡ Quick Commands

```bash
# Essential validation
python tests/demo/strategy_config_demo.py
cd tests && ./lint.sh
python -m pytest tests/unit/ -n auto -x --tb=line

# Development workflow
cd tests && ./lint.sh --test    # Full quality check + tests
pytest -n auto tests/unit/ -v   # Parallel unit tests
pytest -n 0 tests/integration/ -v # Serial integration tests
```

## 🎭 Interactive Demos

**See [`../demo/README.md`](../demo/README.md) for runnable examples:**

- Strategy-based client configuration demo
- Mock data generation showcase
- Test failure analysis patterns

---

**📖 Historical Documentation:**

- `refactoring_for_testability.md` - Production code change log
