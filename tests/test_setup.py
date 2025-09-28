"""
Basic test to verify pytest setup is working correctly.
"""

from tests.common import pytest


def test_pytest_working():
    """Test that pytest is working correctly."""
    assert 1 == 1


def test_fixtures_available(mock_strategy_configs):
    """Test that global fixtures are available."""
    assert "trust" in mock_strategy_configs
    assert mock_strategy_configs["trust"]["aggregation_strategy_keyword"] == "trust"


@pytest.mark.unit
def test_unit_marker():
    """Test that unit test marker works."""
    assert 1 + 1 == 2


class TestSetupValidation:
    """Test class to verify pytest class discovery."""

    def test_class_discovery(self):
        """Test that pytest can discover test classes."""
        assert 2 + 2 == 4

    def test_fixture_injection(self, mock_strategy_configs):
        """Test fixture injection in test classes."""
        assert mock_strategy_configs["pid"]["aggregation_strategy_keyword"] == "pid"
