# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Test temperature (KNMI) functions."""

import json
from unittest.mock import MagicMock

import pytest
import responses
from numpy.testing import assert_almost_equal

from sundael.temperature import (
    _get_knmi_hourly_temperature_series,
    get_knmi_hourly_temperature_series,
    load_temperature_data,
    read_knmi_csv,
)


@pytest.fixture()
def example_csv() -> str:
    """Example KNMI CSV output."""
    return "tests/data/temperature/example.csv"


def test_read_knmi_csv_shape(example_csv: str) -> None:
    """Test if DataFrame has the correct shape."""
    df = read_knmi_csv(example_csv)
    assert df.shape == (48, 3)


def test_read_knmi_csv_columns(example_csv: str) -> None:
    """Test if DataFrame has the correct columns."""
    df = read_knmi_csv(example_csv)
    assert df.columns == ["station_code", "datetime", "temperature"]


def test_read_knmi_csv_temperature(example_csv: str) -> None:
    """Test if DataFrame has the correct temperature values."""
    df = read_knmi_csv(example_csv)
    assert_almost_equal(df["temperature"].max(), 15.5, decimal=1)
    assert_almost_equal(df["temperature"].min(), 4.5, decimal=1)


@pytest.mark.parametrize("station_ids,fmt", (((260,), "json"), (("210", "230"), "csv")))
def test_get_knmi_hourly_temperature_series(station_ids: tuple[str | int], fmt: str) -> None:
    """Test if KNMI API is called correctly."""
    url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(
            responses.POST,
            url,
            body="(260,)",
            status=200,
            content_type="application/json",
            match=[
                responses.matchers.json_params_matcher(
                    {"stns": "260", "vars": "TEMP", "start": "2011010100", "end": "2011020100", "fmt": "json"}
                )
            ],
        )
        rsps.add(
            responses.POST,
            url,
            body="('210', '230')",
            status=200,
            content_type="application/json",
            match=[
                responses.matchers.json_params_matcher(
                    {"stns": "210:230", "vars": "TEMP", "start": "2011010100", "end": "2011020100", "fmt": "csv"}
                )
            ],
        )
        result = get_knmi_hourly_temperature_series("20110101", "20110201", station_ids=station_ids, fmt=fmt)

    assert result == str(station_ids)


@pytest.mark.parametrize("status_code", (408, 429, 500, 502, 503, 504))
def test_get_knmi_hourly_temperature_series_retry(status_code: int) -> None:
    """Test if KNMI API is called correctly with retry."""
    url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(
            responses.POST,
            url,
            body="FALSE",
            status=status_code,
            content_type="application/json",
        )
        rsps.add(
            responses.POST,
            url,
            body="FALSE",
            status=status_code,
            content_type="application/json",
        )
        rsps.add(
            responses.POST,
            url,
            body="(260,)",
            status=200,
            content_type="application/json",
            match=[
                responses.matchers.json_params_matcher(
                    {"stns": "260", "vars": "TEMP", "start": "2011010100", "end": "2011020100", "fmt": "json"}
                )
            ],
        )
        result = get_knmi_hourly_temperature_series("20110101", "20110201", station_ids=(260,), fmt="json")

    assert result == str((260,))


def test_load_temperature_data_year(mocker: MagicMock) -> None:
    """Test load_temperature_data() with year (int) as input."""
    ret = {"date": ["20230101", "20230101"], "station_code": [260, 260], "hour": [1, 2], "T": [10, 11]}
    mocked = mocker.MagicMock(return_value=json.dumps(ret))
    mocker.patch("sundael.temperature.get_knmi_hourly_temperature_series", mocked)
    assert load_temperature_data(2023).shape == (2, 3)


def test_load_temperature_data_provided_file() -> None:
    """Test load_temperature_data() with file in data directory as input."""
    assert load_temperature_data("2023.KNMI.temperature.csv").shape == (8760, 3)


def test_load_temperature_data_with_file() -> None:
    """Test load_temperature_data() with file as input."""
    assert load_temperature_data("tests/data/temperature/example.csv").shape == (48, 3)


def test_load_temperature_data_file_not_exists() -> None:
    """Test load_temperature_data() with file as input."""
    with pytest.raises(FileNotFoundError):
        load_temperature_data("i_do_not_exist.csv")


def test_get_knmi_hourly_temperature_series_uncached_function() -> None:
    """Test direct KNMI helper path without joblib cache interaction."""
    url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(
            responses.POST,
            url,
            body="ok",
            status=200,
            content_type="application/json",
            match=[
                responses.matchers.json_params_matcher(
                    {
                        "stns": "260",
                        "vars": "TEMP",
                        "start": "2011010100",
                        "end": "2011010200",
                        "fmt": "json",
                    }
                )
            ],
        )
        result = _get_knmi_hourly_temperature_series("20110101", "20110102")

    assert result == "ok"


def test_get_knmi_hourly_temperature_series_uncached_function_error() -> None:
    """Test direct KNMI helper raises OSError for non-OK API response."""

    class ResponseMock:
        ok = False

        def __str__(self) -> str:
            return "mocked-response"

    session = MagicMock()
    session.post.return_value = ResponseMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("sundael.temperature.requests.Session", lambda: session)
        with pytest.raises(OSError, match="Accessing KNMI API failed"):
            _get_knmi_hourly_temperature_series("20110101", "20110102")
