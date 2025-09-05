"""
Basic test to verify pytest setup is working correctly.
"""

import pytest


def test_pytest_working():
    """Test that pytest is working correctly."""
    assert True


def test_fixtures_available(sample_trust_config):
    """Test that global fixtures are available."""
    assert "aggregation_strategy_keyword" in sample_trust_config
    assert sample_trust_config["aggregation_strategy_keyword"] == "trust"


@pytest.mark.unit
def test_unit_marker():
    """Test that unit test marker works."""
    assert 1 + 1 == 2


class TestSetupValidation:
    """Test class to verify pytest class discovery."""

    def test_class_discovery(self):
        """Test that pytest can discover test classes."""
        assert True

    def test_fixture_injection(self, sample_pid_config):
        """Test fixture injection in test classes."""
        assert sample_pid_config["aggregation_strategy_keyword"] == "pid"
