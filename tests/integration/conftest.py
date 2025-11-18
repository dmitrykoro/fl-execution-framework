"""
Pytest configuration and fixtures for API integration tests.

Provides:
- api_client: FastAPI TestClient fixture for integration testing
"""

import warnings

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def api_client() -> TestClient:
    """Create a FastAPI TestClient for the app."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx")
        return TestClient(app)
