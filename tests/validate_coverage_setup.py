#!/usr/bin/env python3
"""
Validation script for pytest test suite setup
Tests that core testing components are properly configured
"""

from pathlib import Path
from tests.common import init_test_environment

logger = init_test_environment()


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report status"""
    if Path(filepath).exists():
        logger.info(f"✅ {description}: {filepath}")
        return True
    else:
        logger.error(f"❌ {description}: {filepath} (missing)")
        return False


def check_core_config() -> bool:
    """Validate core test configuration files"""
    logger.info("🔍 Checking core test configuration...")

    files_ok = True
    files_ok &= check_file_exists(".coveragerc", "Coverage config")
    files_ok &= check_file_exists("pytest.ini", "Pytest config")
    files_ok &= check_file_exists("requirements.txt", "Requirements")
    files_ok &= check_file_exists(".github/workflows/ci.yml", "CI workflow")

    return files_ok


def check_pytest_config() -> bool:
    """Check pytest configuration"""
    logger.info("\n🔍 Checking pytest configuration...")

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
                logger.error(f"❌ Missing pytest options: {missing_options}")
            if missing_markers:
                logger.error(f"❌ Missing pytest markers: {missing_markers}")
            return False
        else:
            logger.info("✅ Pytest configuration is complete")
            return True

    except FileNotFoundError:
        logger.error("❌ pytest.ini not found")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading pytest.ini: {e}")
        return False


def check_coverage_rc() -> bool:
    """Check .coveragerc configuration"""
    logger.info("\n🔍 Checking .coveragerc configuration...")

    try:
        with open(".coveragerc", "r", encoding="utf-8") as f:
            content = f.read()

        required_sections = ["[run]", "[report]", "[html]"]
        required_settings = ["source = src", "show_missing = True"]

        missing_sections = [sec for sec in required_sections if sec not in content]
        missing_settings = [set for set in required_settings if set not in content]

        if missing_sections or missing_settings:
            if missing_sections:
                logger.error(f"❌ Missing .coveragerc sections: {missing_sections}")
            if missing_settings:
                logger.error(f"❌ Missing .coveragerc settings: {missing_settings}")
            return False
        else:
            logger.info("✅ .coveragerc configuration is complete")
            return True

    except FileNotFoundError:
        logger.error("❌ .coveragerc not found")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading .coveragerc: {e}")
        return False


def check_dependencies() -> bool:
    """Check required dependencies"""
    logger.info("\n🔍 Checking dependencies...")

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
            logger.info(f"✅ {display_name}: {version}")
        except ImportError:
            logger.error(f"❌ {display_name} not installed")
            all_installed = False

    return all_installed


def check_ci_config() -> bool:
    """Check CI configuration"""
    logger.info("\n🔍 Checking CI configuration...")

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
            logger.error(f"❌ Missing CI elements: {missing_elements}")
            return False
        else:
            logger.info("✅ CI configuration is complete")
            return True

    except FileNotFoundError:
        logger.error("❌ CI workflow not found")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading CI config: {e}")
        return False


def main() -> int:
    """Main validation function"""
    logger.info("🧪 Pytest Test Suite Validation")
    logger.info("=" * 40)

    # Run all checks
    checks: list[tuple[str, bool]] = [
        ("Core config files", check_core_config()),
        ("Pytest configuration", check_pytest_config()),
        ("Coverage configuration", check_coverage_rc()),
        ("Dependencies", check_dependencies()),
        ("CI configuration", check_ci_config()),
    ]

    # Summary
    logger.info("\n📋 Validation Summary:")
    logger.info("-" * 25)

    passed = sum(result for _, result in checks)
    total = len(checks)

    for check_name, result in checks:
        status = "✅" if result else "❌"
        if result:
            logger.info(f"{status} {check_name}")
        else:
            logger.error(f"{status} {check_name}")

    if passed == total:
        logger.info(f"\n🎉 All checks passed ({passed}/{total})")
        logger.info("\nNext steps:")
        logger.info("  1. Run: pytest --cov=src")
        logger.info("  2. Check: htmlcov/index.html")
        logger.info("  3. Commit and push to trigger CI")
        return 0
    else:
        logger.warning(f"\n🔧 Some checks failed ({passed}/{total})")
        logger.warning("Please fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
