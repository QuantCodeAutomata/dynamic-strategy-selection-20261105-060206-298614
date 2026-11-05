"""Shared pytest fixtures and marker registration.

The ``market_data`` marker is used by tests that need a reduced slice of real
market data loaded through ``data/data_loader.py``; they are skipped when
``DATALOADER_API_KEY`` is not set in the environment.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that load a reduced slice of real market data"
    )


def _has_data_keys() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def market_data_slice(request: pytest.FixtureRequest):
    """A small, cached slice of French factors and FRED series for data-path tests."""
    if not _has_data_keys():
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set: skipping market_data test")
    cache = request.config.cache
    cached = cache.get("market_data_slice/french", None)
    if cached is not None:
        import pandas as pd
        french = pd.read_json(cached)
        french.index = pd.PeriodIndex(french.index, freq="M")
        return french

    from data.data_loader import load_french_factors
    french = load_french_factors()
    cache.set("market_data_slice/french", french.to_json())
    return french


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
