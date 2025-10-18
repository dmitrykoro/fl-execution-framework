#!/usr/bin/env python3
"""
Launcher for the mock data showcase with Unicode support.

Usage:
  python -m tests.demo.run_showcase
"""

from tests.common import init_test_environment
from tests.demo.mock_data_showcase import TestMockDataShowcase

logger = init_test_environment(include_timestamp=True)

if __name__ == "__main__":
    import sys

    logger.info("🎭 Starting Mock Data Showcase...")
    logger.info(f"Platform: {sys.platform}")
    logger.info("=" * 50)

    showcase = TestMockDataShowcase()
    showcase.test_run_all_demonstrations()
