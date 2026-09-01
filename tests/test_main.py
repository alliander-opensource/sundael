# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Test functions for main functionality."""

import numpy as np
import pandas as pd
import pytest

from sundael import PVDisaggregator


@pytest.fixture
def datetimes() -> pd.Series:
    """Test datetimes series fixture."""
    return pd.to_datetime(["2023-01-19 14:00", "2023-07-28 12:00", "2023-10-07 16:00"], utc=True)


@pytest.fixture
def pv_object(datetimes: pd.Series) -> PVDisaggregator:
    """PVDisaggregator fixture."""
    pv = PVDisaggregator(pv_ratio=pd.DataFrame(columns=pd.DatetimeIndex([])))
    pv.time_stamps = datetimes
    pv.temperature = np.array([0.0] * len(pv.time_stamps))
    return pv


def test_pv_object_fixture(pv_object: PVDisaggregator) -> None:
    """Ensure the fixture can be instantiated and has expected shape."""
    assert len(pv_object.time_stamps) == 3
    assert pv_object.temperature.shape == (3,)


def test_pvdisaggregator_import_and_disaggregate_smoke() -> None:
    """Smoke test for public import path and disaggregate return keys."""
    datetimes = pd.date_range("2023-01-01", periods=96, freq="15min", tz="UTC")
    pv_ratio = pd.DataFrame(np.ones((len(datetimes), 1)), index=datetimes, columns=["default"])
    netload = pd.DataFrame([np.linspace(-1.0, 1.0, len(datetimes))], index=["customer1"], columns=datetimes)

    pv = PVDisaggregator(
        pv_ratio=pv_ratio,
        area_mapping={"customer1": "default"},
        temperature=np.zeros(len(datetimes)),
    )

    params = pd.DataFrame(
        [[0.0, 1.0, 0.65, 0.0, 0.5]],
        index=["customer1"],
        columns=["base", "effective_size", "tilt", "orientation", "c_temp"],
    )
    generation = pd.DataFrame([np.ones(len(datetimes))], index=["customer1"], columns=datetimes)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(PVDisaggregator, "optimize_parameters", lambda self, _: params)
        monkeypatch.setattr(PVDisaggregator, "estimate_generation", lambda self, *_: generation)
        result = pv.disaggregate(net_load=netload, include_self_consumption=False)

    assert set(result) == {"params", "net_con", "net_gen", "generation", "consumption"}
