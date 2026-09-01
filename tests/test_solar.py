# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Test functions for solar functions."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from sundael.config import DEFAULT_CONFIG, PVConfig
from sundael.solar import SolarProperties, algf, estimate_parameters, solar_maxgen


@pytest.mark.parametrize(
    "params,expected_result",
    (({}, [574.1718, 1487.5203, 264.9181]), ({"tilt": 0.0, "orientation": 1.5}, [229.4306, 1269.5863, 154.2824])),
)
def test_solar_maxgen(params: dict[str, float], expected_result: list[float]) -> None:
    """Ensure solar_maxgen output is correct (based on R implementation)."""
    # Result values taken from R implementation, with same input
    altitude = np.array([0.2123017, 0.9894110, 0.1579230])
    azimuth = np.array([0.53199170, 0.08928495, 1.20001719])
    irradiance = np.array([0.4922709, 0.9045511, 0.3997977])

    call_kwargs: dict[str, object] = {
        "cos_altitude": np.cos(altitude),
        "sin_altitude": np.sin(altitude),
        "azimuth": azimuth,
        "irradiance": irradiance,
        "effective_size": 1500,
        "tilt": DEFAULT_CONFIG.default_tilt,
        "orientation": DEFAULT_CONFIG.default_orientation,
        "c_temp": DEFAULT_CONFIG.default_c_temp,
        "temperature": 0,
    }
    call_kwargs.update(params)
    result = solar_maxgen.py_func(**call_kwargs)
    np.testing.assert_allclose(result, expected_result, atol=1e-4)


@pytest.mark.parametrize(
    "w,error,expected",
    (([0.97623183, 0.99555505, 0.94762338], 1, 1431.51), ([0.97623183, 0.99555505, 0.94762338], 2, 1466.67)),
)
def test_algf(w: list[float], error: float, expected: float) -> None:
    """Test error function."""
    params = (100, 2000, 0.65, 0.1, 0)
    netload = np.array([-400, 2000, -300])
    n_time_stamps = 3
    altitude = np.array([1.01917207, 1.04057071, 0.98839471])
    azimuth = np.array([-0.36976409, 0.10281166, 0.55468109])
    irradiance = np.array([0.90919765, 0.91233255, 0.90438633])

    result = algf.py_func(
        params=params,
        w=w,
        netload=netload,
        n_time_stamps=n_time_stamps,
        error=error,
        cos_altitude=np.cos(altitude),
        sin_altitude=np.sin(altitude),
        azimuth=azimuth,
        irradiance=irradiance,
        temperature=DEFAULT_CONFIG.yearly_avg_temp,
    )

    assert_almost_equal(result, expected, decimal=2)


def test_estimate_parameters() -> None:
    """Test parameter estimation function used in optimization."""
    netload = pd.Series([-400, 2000, -300])
    altitude = np.array([1.01917207, 1.04057071, 0.98839471])
    azimuth = np.array([-0.36976409, 0.10281166, 0.55468109])
    irradiance = np.array([0.90919765, 0.91233255, 0.90438633])
    r_avg = np.array([0.97623183, 0.99555505, 0.94762338])

    expected = [5.721467e-06, 2.016941e03, 5.548540e-01, 1.667794e-01, 9.399049e-02]
    params = estimate_parameters(
        netload=netload,
        altitude=altitude,
        azimuth=azimuth,
        irradiance=irradiance,
        r_avg=r_avg,
        temperature=DEFAULT_CONFIG.yearly_avg_temp,
    )

    np.testing.assert_allclose(params, expected, rtol=0.01, atol=10)


@pytest.mark.parametrize("temp_low,temp_high", ((0.0, 10.0), (10.0, 30.0), (0.0, 30.0)))
def test_solar_maxgen_temperature(temp_low: float, temp_high: float) -> None:
    """Assert that generation is lower at higher temperatures."""
    altitude = np.array([0.2123017, 0.9894110, 0.1579230])
    azimuth = np.array([0.53199170, 0.08928495, 1.20001719])
    irradiance = np.array([0.4922709, 0.9045511, 0.3997977])

    result_low = solar_maxgen(
        cos_altitude=np.cos(altitude),
        sin_altitude=np.sin(altitude),
        azimuth=azimuth,
        irradiance=irradiance,
        effective_size=1500,
        tilt=DEFAULT_CONFIG.default_tilt,
        orientation=DEFAULT_CONFIG.default_orientation,
        c_temp=DEFAULT_CONFIG.default_c_temp,
        temperature=temp_low,
    )
    result_high = solar_maxgen(
        cos_altitude=np.cos(altitude),
        sin_altitude=np.sin(altitude),
        azimuth=azimuth,
        irradiance=irradiance,
        effective_size=1500,
        tilt=DEFAULT_CONFIG.default_tilt,
        orientation=DEFAULT_CONFIG.default_orientation,
        c_temp=DEFAULT_CONFIG.default_c_temp,
        temperature=temp_high,
    )
    assert np.all(result_high < result_low)


@pytest.fixture
def datetimes() -> pd.Series:
    """Test datetimes series fixture."""
    return pd.to_datetime(["2023-01-19 14:00", "2023-07-28 12:00", "2023-10-07 16:00"], utc=True)


@pytest.fixture
def solar_object(datetimes: pd.Series) -> SolarProperties:
    """PVDisaggregator fixture."""
    return SolarProperties(time_stamps=datetimes, temperature=np.array([0.0] * len(datetimes)))


def test_azimuth(solar_object: SolarProperties) -> None:
    """Test azimuth method of SolarProperties."""
    # Result values checked with https://gml.noaa.gov/grad/solcalc/
    np.testing.assert_allclose(solar_object.azimuth, [0.53409, 0.09186, 1.2004], atol=1e-4)


def test_altitude(solar_object: SolarProperties) -> None:
    """Test altitude method of SolarProperties."""
    # Result values checked with https://gml.noaa.gov/grad/solcalc/
    np.testing.assert_allclose(solar_object.altitude, [0.2143, 0.9878, 0.1560], atol=1e-4)


def test_air_mass(solar_object: SolarProperties) -> None:
    """Test air_mass method of SolarProperties."""
    # Originally result values from R implementation.
    # Updated to based on pvlib result values, still close to values of R implementation.
    np.testing.assert_allclose(solar_object.air_mass, [4.609489, 1.197331, 6.195723], atol=1e-5)


def test_irradiance(solar_object: SolarProperties) -> None:
    """Test irradiance method of SolarProperties."""
    # Result values from R implementation
    np.testing.assert_allclose(solar_object.irradiance, [0.495186, 0.904229, 0.396137], atol=1e-5)


def test_reference_average(solar_object: SolarProperties) -> None:
    """Test reference_average function of SolarProperties."""
    # Result values from R implementation
    np.testing.assert_allclose(solar_object.reference_average, [0.3854, 0.991353, 0.174355], atol=1e-5)


def test_solar_maxgen_with_array_temperature_and_negative_altitude() -> None:
    """Cover array temperature path and skip branch for sun below horizon."""
    altitude = np.array([-0.1, 0.7])
    azimuth = np.array([0.1, 0.2])
    irradiance = np.array([1.0, 1.0])
    temperature = np.array([10.0, 10.0])

    result = solar_maxgen.py_func(
        cos_altitude=np.cos(altitude),
        sin_altitude=np.sin(altitude),
        azimuth=azimuth,
        irradiance=irradiance,
        effective_size=1000,
        tilt=DEFAULT_CONFIG.default_tilt,
        orientation=DEFAULT_CONFIG.default_orientation,
        c_temp=DEFAULT_CONFIG.default_c_temp,
        temperature=temperature,
    )

    assert result[0] == 0
    assert result[1] > 0


def test_solar_properties_uses_config_defaults(datetimes: pd.Series) -> None:
    """Ensure SolarProperties reads defaults from custom config."""
    config = PVConfig(latitude=51.5, longitude=5.0, yearly_avg_temp=9.0)
    solar = SolarProperties(time_stamps=datetimes, config=config)

    assert solar.latitude == 51.5
    assert solar.longitude == 5.0
    np.testing.assert_allclose(solar.temperature, np.array([9.0] * len(datetimes)))
