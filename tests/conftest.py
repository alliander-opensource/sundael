# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Shared test configuration."""

from collections.abc import Generator
from typing import NoReturn

import pytest


@pytest.fixture(autouse=True)
def prevent_unmocked_knmi_api(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Fail fast when application tests attempt an unmocked KNMI request."""

    def fail_on_knmi_request(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Tests must mock calls to the KNMI API")

    monkeypatch.setattr("sundael.temperature.get_knmi_hourly_temperature_series", fail_on_knmi_request)
    monkeypatch.setattr("sundael.disaggregation.get_knmi_hourly_temperature_series", fail_on_knmi_request)
    yield
