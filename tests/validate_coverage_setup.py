#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for pytest test suite setup
Tests that core testing components are properly configured
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Set UTF-8 encoding for Windows
if os.name == "nt":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Reconfigure stdout/stderr for UTF-8
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report status"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (missing)")
        return False


def check_core_config() -> bool:
    """Validate core test configuration files"""
    print("🔍 Checking core test configuration...")

    files_ok = True
    files_ok &= check_file_exists(".coveragerc", "Coverage config")
    files_ok &= check_file_exists("pytest.ini", "Pytest config")
    files_ok &= check_file_exists("requirements.txt", "Requirements")
    files_ok &= check_file_exists(".github/workflows/ci.yml", "CI workflow")

    return files_ok


def check_pytest_config() -> bool:
    """Check pytest configuration"""
    print("\n🔍 Checking pytest configuration...")

    try:
        with open("pytest.ini", "r", encoding="utf-8") as f:
            content = f.read()

        required_options = [
            "--cov=src",
            "--cov-report=term-missing",
        ]

        required_markers = [
            "unit: Unit tests",
            "integration: Integration tests",
            "tutorial: Tutorial tests",
        ]

        missing_options = [opt for opt in required_options if opt not in content]
        missing_markers = [mark for mark in required_markers if mark not in content]

        if missing_options or missing_markers:
            if missing_options:
                print(f"❌ Missing pytest options: {missing_options}")
            if missing_markers:
                print(f"❌ Missing pytest markers: {missing_markers}")
            return False
        else:
            print("✅ Pytest configuration is complete")
            return True

    except FileNotFoundError:
        print("❌ pytest.ini not found")
        return False
    except Exception as e:
        print(f"❌ Error reading pytest.ini: {e}")
        return False


def check_coverage_rc() -> bool:
    """Check .coveragerc configuration"""
    print("\n🔍 Checking .coveragerc configuration...")

    try:
        with open(".coveragerc", "r", encoding="utf-8") as f:
            content = f.read()

        required_sections = ["[run]", "[report]", "[html]"]
        required_settings = ["source = src", "show_missing = True"]

        missing_sections = [sec for sec in required_sections if sec not in content]
        missing_settings = [set for set in required_settings if set not in content]

        if missing_sections or missing_settings:
            if missing_sections:
                print(f"❌ Missing .coveragerc sections: {missing_sections}")
            if missing_settings:
                print(f"❌ Missing .coveragerc settings: {missing_settings}")
            return False
        else:
            print("✅ .coveragerc configuration is complete")
            return True

    except FileNotFoundError:
        print("❌ .coveragerc not found")
        return False
    except Exception as e:
        print(f"❌ Error reading .coveragerc: {e}")
        return False


def check_dependencies() -> bool:
    """Check required dependencies"""
    print("\n🔍 Checking dependencies...")

    dependencies = [
        ("pytest", "pytest"),
        ("coverage", "coverage"),
        ("pytest_cov", "pytest-cov"),
    ]

    all_installed = True
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {display_name}: {version}")
        except ImportError:
            print(f"❌ {display_name} not installed")
            all_installed = False

    return all_installed


def check_ci_config() -> bool:
    """Check CI configuration"""
    print("\n🔍 Checking CI configuration...")

    try:
        with open(".github/workflows/ci.yml", "r", encoding="utf-8") as f:
            ci_content = f.read()

        required_elements = [
            "pytest",
            "--cov=src",
            "--cov-report=xml",
            "--cov-report=term-missing",
        ]

        missing_elements = [
            elem for elem in required_elements if elem not in ci_content
        ]

        if missing_elements:
            print(f"❌ Missing CI elements: {missing_elements}")
            return False
        else:
            print("✅ CI configuration is complete")
            return True

    except FileNotFoundError:
        print("❌ CI workflow not found")
        return False
    except Exception as e:
        print(f"❌ Error reading CI config: {e}")
        return False


def main() -> int:
    """Main validation function"""
    print("🧪 Pytest Test Suite Validation")
    print("=" * 40)

    # Run all checks
    checks: List[Tuple[str, bool]] = [
        ("Core config files", check_core_config()),
        ("Pytest configuration", check_pytest_config()),
        ("Coverage configuration", check_coverage_rc()),
        ("Dependencies", check_dependencies()),
        ("CI configuration", check_ci_config()),
    ]

    # Summary
    print("\n📋 Validation Summary:")
    print("-" * 25)

    passed = sum(result for _, result in checks)
    total = len(checks)

    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")

    if passed == total:
        print(f"\n🎉 All checks passed ({passed}/{total})")
        print("\nNext steps:")
        print("  1. Run: pytest --cov=src")
        print("  2. Check: htmlcov/index.html")
        print("  3. Commit and push to trigger CI")
        return 0
    else:
        print(f"\n🔧 Some checks failed ({passed}/{total})")
        print("Please fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
