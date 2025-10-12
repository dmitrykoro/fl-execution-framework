"""Shared fixtures for output handler tests."""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_matplotlib_base():
    """Mock base matplotlib pyplot functions."""
    with (
        patch("matplotlib.pyplot.figure") as mock_figure,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
    ):
        yield {
            "figure": mock_figure,
            "show": mock_show,
            "savefig": mock_savefig,
        }


@pytest.fixture
def mock_matplotlib_full():
    """Mock full matplotlib pyplot functions including plot/bar/legend."""
    with (
        patch("matplotlib.pyplot.figure") as mock_figure,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.plot") as mock_plot,
        patch("matplotlib.pyplot.bar") as mock_bar,
        patch("matplotlib.pyplot.legend") as mock_legend,
        patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
        patch("matplotlib.pyplot.subplots") as mock_subplots,
    ):
        # Setup subplots return value
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        yield {
            "figure": mock_figure,
            "show": mock_show,
            "savefig": mock_savefig,
            "plot": mock_plot,
            "bar": mock_bar,
            "legend": mock_legend,
            "tight_layout": mock_tight_layout,
            "subplots": mock_subplots,
            "mock_fig": mock_fig,
            "mock_axes": mock_axes,
        }
