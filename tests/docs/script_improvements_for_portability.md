# Script Improvements for Cross-Platform Portability & Best Practices

## 📜 Summary

The project's shell scripts (`run_simulation.sh`, `reinstall_requirements.sh`, and `tests/lint.sh`) have been enhanced to follow software engineering best practices for cross-platform compatibility, robustness, and maintainability. These changes improve the developer experience across Windows, macOS, and Linux environments.

## 🎯 The Purpose of Script Improvements

Modern research software needs to work reliably across different development environments:

- **Cross-platform compatibility** for diverse research teams
- **Robust error handling** to prevent silent failures
- **Multiple Python version support** for different environments
- **Better user experience** with clear feedback and automatic fallbacks
- **Professional software engineering standards** for research reproducibility

---

## 🛠️ Summary of Script Modifications

### 1. Multi-Python Version Support 🐍

- **Change:** Extended Python detection from hardcoded `python3.10` to support Python 3.9+
- **Purpose:** Accommodates different environments where various Python versions are available
- **Implementation:** Priority-ordered detection: `python3.11` → `python3.10` → `python3.9` → `python3` → `python`

#### Multi-Python Support Details

- **Affected Scripts:** `run_simulation.sh`, `reinstall_requirements.sh`
- **Before:** `if command_exists python3.10; then PYTHON=python3.10`
- **After:** Loop through version list with `sys.version_info >= (3, 9)` validation
- **Detection Order:** `python` → `python3.11` → `python3.10` → `python3.9` → `python3` → Windows `py` launcher
- **Benefit:** Works on systems with different Python installations
- **Risk Assessment:** Very low - maintains backward compatibility while adding flexibility

---

### 2. Windows Cross-Platform Support 🪟

- **Change:** Added Windows `py` launcher fallback and Scripts/activate detection
- **Purpose:** Enables the framework to work on Windows environments using MINGW64/Git Bash
- **Implementation:** Detects and uses Windows-specific paths and activation scripts

#### Windows Support Details

- **Affected Scripts:** `run_simulation.sh`, `reinstall_requirements.sh`, `tests/lint.sh`
- **Windows Detection:**
  - `py -3.11`, `py -3.10`, `py -3.9` launcher support
  - `$VENV_DIR/Scripts/activate` vs `$VENV_DIR/bin/activate` path handling
  - Dynamic virtual environment detection for both `.venv` and `venv`
- **Benefit:** Seamless operation on Windows development environments
- **Risk Assessment:** Very low - adds Windows support without affecting Unix systems

---

### 3. Enhanced Error Handling & Robustness 🛡️

- **Change:** Added `set -e` (fail-fast) and improved error validation
- **Purpose:** Prevents silent failures and provides clear error messages
- **Implementation:** Early exit on any command failure with descriptive error messages

#### Error Handling Details

- **Affected Scripts:** All shell scripts
- **Before:** Scripts could continue after errors
- **After:** `set -e` ensures immediate exit on any failure
- **Additional Checks:** Version validation, path verification, tool availability
- **Benefit:** Faster debugging and more reliable script execution
- **Risk Assessment:** Very low - standard shell scripting best practice

---

### 4. Flexible Virtual Environment Handling 📁

- **Change:** Dynamic venv detection supporting both `.venv` and `venv` naming conventions
- **Purpose:** Accommodates developer preferences while supporting modern Python development conventions
- **Implementation:** Smart detection with backward compatibility and modern defaults

#### Virtual Environment Details

- **Affected Scripts:** `run_simulation.sh`, `reinstall_requirements.sh`, `tests/lint.sh`
- **Detection Logic:**
  - `reinstall_requirements.sh`: Checks `venv` → `.venv` → defaults to `.venv` (preserves existing)
  - `run_simulation.sh`: Checks `.venv` → `venv` → creates `.venv` (modern preference)
  - `tests/lint.sh`: Checks `.venv` → `venv` with activation tracking
- **Activation:** Cross-platform detection of activation scripts (`Scripts/activate` vs `bin/activate`)
- **Benefit:** Works with both traditional `venv` and modern `.venv` naming conventions
- **Risk Assessment:** Very low - maintains full backward compatibility while supporting modern practices

---

### 5. Robust Download Mechanisms ⬇️

- **Change:** Added Python urllib fallback when wget is unavailable
- **Purpose:** Ensures dataset download works on systems without wget (common on Windows)
- **Implementation:** Detect wget availability, fallback to Python's built-in urllib

#### Download Mechanism Details

- **File:** `run_simulation.sh`
- **Primary:** `wget` for Linux/macOS
- **Fallback:** `python -c "import urllib.request; urllib.request.urlretrieve(...)"` for Windows
- **Benefit:** Universal dataset download capability
- **Risk Assessment:** Very low - standard approach for cross-platform downloads

---

### 6. Module Execution Improvements 🔧

- **Change:** Updated Python execution from direct script to module execution
- **Purpose:** Follows Python best practices and ensures proper import paths
- **Implementation:** `python src/simulation_runner.py` → `python -m src.simulation_runner`

#### Module Execution Details

- **File:** `run_simulation.sh`
- **Purpose:** Ensures Python module imports work correctly regardless of working directory
- **Benefit:** More robust execution that follows Python packaging standards
- **Risk Assessment:** Very low - standard Python best practice

---

### 7. Enhanced Development Tooling 🔍

- **Change:** Added comprehensive `tests/lint.sh` with modern tools (ruff, pyright, mypy)
- **Purpose:** Provides professional-grade code quality tools for research software
- **Implementation:** Replaces older tools with modern, faster alternatives

#### Development Tooling Details

- **File:** `tests/lint.sh`
- **Virtual Environment:** Dynamic detection supporting both `.venv` and `venv`
- **Modern Tools:**
  - **ruff**: Replaces `flake8` + `isort` + `black` (10-100x faster)
  - **pyright**: Additional type checking beyond mypy
  - **pytest**: Comprehensive testing
- **Modes:** Basic (quick checks) vs Full (complete validation)
- **Activation Tracking:** Uses `VENV_ACTIVATED` flag for cleaner logic
- **Benefit:** Faster, more comprehensive code quality checking with better environment detection
- **Risk Assessment:** Very low - adds developer productivity tools

---

## 📈 Summary of Impact

### ✅ **Key Outcomes:**

1. 🌍 **Cross-platform compatibility** - Works on Windows, macOS, and Linux
2. 🛡️ **Improved reliability** - Better error handling and validation
3. 🐍 **Flexible Python support** - Works with Python 3.9+ versions
4. 🚀 **Better developer experience** - Clear feedback and automatic fallbacks
5. 📊 **Professional tooling** - Modern code quality and testing tools

### 📝 **Developer Guidance:**

1. 🐍 **Python Requirements**: Scripts now support Python 3.9+ (previously required exactly 3.10)
2. 🪟 **Windows Support**: Full compatibility with Windows development environments
3. 📁 **Virtual Environments**: Supports both `.venv` (modern) and `venv` (traditional) naming conventions
4. 🔧 **Error Handling**: Scripts fail fast with clear error messages
5. 🛠️ **Tooling**: Enhanced `tests/lint.sh` with dynamic environment detection

### 🚀 **Recommended Workflow:**

1. ✅ **Cross-platform development**: Scripts work consistently across all platforms
2. 🛡️ **Reliable execution**: Enhanced error handling catches issues early
3. 🚀 **Modern tooling**: Use `tests/lint.sh` for code quality checks

---

## ❓ FAQ

**Q: Will existing workflows be affected?**
A: No. All changes are backward compatible while adding new capabilities.

**Q: Do I need to change my Python version?**
A: No, but the scripts now support Python 3.9+ instead of requiring exactly 3.10.

**Q: What was the reason for these script improvements?**
A: To follow modern software engineering best practices and support diverse development environments.

**Q: Are these changes necessary for research work?**
A: Yes - reliable, cross-platform scripts are essential for reproducible research and team collaboration.

**Q: What should I do if a script fails?**
A: The enhanced error handling will provide clear error messages to help debug issues quickly.

---

## 🔧 Technical Implementation Details

### Python Version Detection Logic

```bash
# Priority-ordered detection with version validation
for version in python python3.11 python3.10 python3.9 python3; do
    if command_exists $version; then
        VERSION_CHECK=$($version -c "import sys; print(sys.version_info >= (3, 9))" 2>/dev/null)
        if [ "$VERSION_CHECK" = "True" ]; then
            PYTHON=$version
            break
        fi
    fi
done

# Fallback to Windows py launcher
if [ -z "$PYTHON" ] && command_exists py; then
    for version in "-3.11" "-3.10" "-3.9" "-3"; do
        VERSION_CHECK=$(py $version -c "import sys; print(sys.version_info >= (3, 9))" 2>/dev/null)
        if [ "$VERSION_CHECK" = "True" ]; then
            PYTHON="py $version"
            break
        fi
    done
fi
```

### Dynamic Virtual Environment Detection

```bash
# reinstall_requirements.sh: Preserve existing naming, default to .venv
VENV_DIR=""
if [ -d "venv" ]; then
    VENV_DIR="venv"
elif [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    VENV_DIR=".venv"  # Default to .venv for new installations
fi
```

```bash
# run_simulation.sh: Check .venv first, then venv
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
fi
```

```bash
# tests/lint.sh: Enhanced detection with activation tracking
VENV_ACTIVATED=false
if [ -d ".venv" ]; then
    echo "🔌 Found .venv directory, activating virtual environment..."
    if [ -f ".venv/Scripts/activate" ]; then
        source ".venv/Scripts/activate"
        VENV_ACTIVATED=true
    elif [ -f ".venv/bin/activate" ]; then
        source ".venv/bin/activate"
        VENV_ACTIVATED=true
    fi
elif [ -d "venv" ]; then
    echo "🔌 Found venv directory, activating virtual environment..."
    if [ -f "venv/Scripts/activate" ]; then
        source "venv/Scripts/activate"
        VENV_ACTIVATED=true
    elif [ -f "venv/bin/activate" ]; then
        source "venv/bin/activate"
        VENV_ACTIVATED=true
    fi
fi
```

### Cross-Platform Virtual Environment Activation

```bash
# Handle Windows and Unix-like activation paths
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"  # Windows
else
    source "$VENV_DIR/bin/activate"      # Unix/Linux/macOS
fi
```

### Robust Download with Fallback

```bash
# Use wget for downloads, fall back to Python for portability
if [ "$DOWNLOAD_METHOD" = "wget" ]; then
    wget https://example.com/file.tar
else
    $PYTHON -c "import urllib.request; urllib.request.urlretrieve('https://example.com/file.tar', 'file.tar')"
fi
```

---

## 🏁 Conclusion

These script improvements transform the FL execution framework into a professional, cross-platform research tool. The enhancements follow software engineering best practices while maintaining full backward compatibility.

Key benefits for researchers:

- 🌍 **Universal compatibility** across development environments (Windows, macOS, Linux)
- 🛡️ **Reliable execution** with proper error handling and fail-fast behavior
- � **Flexible virtual environments** supporting both `.venv` and `venv` naming conventions
- 🐍 **Multi-Python support** working with Python 3.9+ versions
- �🚀 **Enhanced productivity** with modern development tooling
- 📊 **Professional standards** for reproducible research

The framework now accommodates diverse developer preferences and environments while maintaining the reliability expected of modern research software. Whether you prefer traditional `venv` directories or modern `.venv` conventions, all scripts work seamlessly.

Happy researching! 🎓🚀
