# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Tests for disaggregation module."""

import contextlib
from collections import namedtuple
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pytest

from sundael.config import DEFAULT_CONFIG
from sundael.disaggregation import (
    N_SAMPLE,
    PVDisaggregator,
    apply_generation_rules,
    calculate_rule_based_self_consumption,
    correct_generation_with_net_gen,
    get_solar_properties,
    run_estimate_parameters,
)
from sundael.utils import mean_gt_zero

EMPTY_RATIO = pd.DataFrame(columns=pd.DatetimeIndex([]))


@pytest.fixture
def net_con() -> npt.NDArray[np.float64]:
    """Net consumption fixture."""
    return np.array(
        [
            [1.0, 1.0, 1.0, 3.0, 0.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 3.0, 0.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )


@pytest.fixture
def consumption() -> npt.NDArray[np.float64]:
    """Consumption fixture."""
    return np.array(
        [
            [2, 2, 1, 8, 2, 0, 4, 2, 2, 2, 2],
            [2, 2, 1, 8, 2, 0, 4, 2, 2, 2, 2],
        ],
        dtype="float64",
    )


def test_basic_consumption_fixtures(net_con: npt.NDArray[np.float64], consumption: npt.NDArray[np.float64]) -> None:
    """Use basic fixtures in a lightweight sanity test."""
    assert net_con.shape == consumption.shape
    assert float(consumption[0, 0] - net_con[0, 0]) == 1.0


def test_set_reference_requires_at_most_one_of_reference_or_ratio() -> None:
    """Ensure supplying both reference and ratio is rejected, while supplying neither is allowed."""
    with pytest.raises(ValueError, match="either a reference or a ratio"):
        PVDisaggregator(
            pv_ratio=pd.DataFrame(index=pd.DatetimeIndex([])), pv_reference=pd.DataFrame(index=pd.DatetimeIndex([]))
        )

    # Supplying neither is valid: a ratio of 1.0 is used for every time stamp.
    pv = PVDisaggregator(pv_ratio=None, pv_reference=None)
    assert not hasattr(pv, "_pv_ratio")


def test_pv_ratio_polars_requires_exactly_one_datetime_column() -> None:
    """Ensure polars PV ratio input validates datetime columns."""
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO)

    with pytest.raises(ValueError, match="DateTime column"):
        pv.pv_ratio = pl.DataFrame({"x": [1.0]})

    dt = [pd.Timestamp("2024-01-01", tz="UTC")]
    with pytest.raises(ValueError, match="exactly one DateTime column"):
        pv.pv_ratio = pl.DataFrame(
            {
                "dt1": dt,
                "dt2": dt,
                "value": [1.0],
            }
        )


def test_pv_ratio_setter_accepts_polars() -> None:
    """Ensure pv_ratio setter handles valid polars input and pandas Series."""
    one_dt = pd.Timestamp("2024-01-01", tz="UTC")
    df = pl.DataFrame({"dt": [one_dt], "value": [1.0]}).with_columns(pl.col("dt").dt.cast_time_unit("ns"))
    pv = PVDisaggregator(pv_ratio=df)
    assert list(pv.pv_ratio.columns) == [one_dt]


def test_pv_ratio_setter_accepts_pd_series() -> None:
    """Ensure pv_ratio setter handles valid polars input and pandas Series."""
    one_dt = pd.Timestamp("2024-01-01", tz="UTC")
    pv = PVDisaggregator(pv_ratio=pd.Series([1.0], index=[one_dt]))
    assert pv.pv_ratio.shape == (1, 1)


def test_pv_ratio_setter_requires_datetime_columns() -> None:
    """Ensure pv_ratio setter rejects inputs without datetime columns."""
    with pytest.raises(ValueError, match="DataFrame needs a DateTime index"):
        PVDisaggregator(pv_ratio=pd.DataFrame([[1.0]], index=["area1"], columns=["not-datetime"]))


def test_area_mapping_required_for_multi_area_ratio() -> None:
    """Ensure area mapping is required when pv_ratio has more than one area."""
    ratio = pd.DataFrame(
        [[1.0, 1.0]],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]),
        columns=["A", "B"],
    )
    with pytest.raises(ValueError, match="Please provide an area mapping"):
        PVDisaggregator(pv_ratio=ratio)


def test_convert_pv_reference_to_ratio_requires_datetime_axis() -> None:
    """Ensure conversion requires either datetime index or columns."""
    with pytest.raises(ValueError, match="DatetimeIndex"):
        PVDisaggregator.convert_pv_reference_to_ratio(pd.DataFrame([[1.0]], index=["a"], columns=["b"]))


def test_convert_pv_reference_to_ratio_from_datetime_index() -> None:
    """Ensure conversion works when datetimes are on the index."""
    dt = pd.date_range("2023-01-01", periods=4, freq="15min", tz="UTC")
    reference = pd.DataFrame({"area1": [0.0, 1.0, 2.0, 1.0]}, index=dt)

    generation = pd.DataFrame([[1.0, 1.0, 2.0, 1.0]], index=["area1"], columns=dt)
    with patch.object(PVDisaggregator, "disaggregate", return_value={"generation": generation}):
        ratio = PVDisaggregator.convert_pv_reference_to_ratio(reference)

    assert ratio.shape == (1, 4)
    assert (ratio.values >= 0).all()
    assert (ratio.values <= 1).all()


def test_convert_pv_reference_to_ratio_from_datetime_columns() -> None:
    """Ensure conversion also works when datetimes are on columns."""
    dt = pd.date_range("2023-01-01", periods=2, freq="15min", tz="UTC")
    reference = pd.DataFrame([[0.0, 1.0]], index=["area1"], columns=dt)

    generation = pd.DataFrame([[1.0, 1.0]], index=["area1"], columns=dt)
    with patch.object(PVDisaggregator, "disaggregate", return_value={"generation": generation}):
        ratio = PVDisaggregator.convert_pv_reference_to_ratio(reference)

    assert ratio.shape == (1, 2)


def test_pvdisaggregator_temperature_unknown_type() -> None:
    """Ensure unsupported temperature value types are rejected."""
    bad_temperature: Any = 42
    with pytest.raises(ValueError, match="Unknown type"):
        PVDisaggregator(pv_ratio=EMPTY_RATIO, temperature=bad_temperature)


def test_temperature_setter_accepts_float() -> None:
    """Ensure scalar temperatures are expanded to the full timestamp length."""
    dt = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    ratio = pd.DataFrame(np.ones((len(dt), 1)), index=dt, columns=["area1"])
    pv = PVDisaggregator(pv_ratio=ratio, temperature=np.zeros(len(dt)))

    pv.temperature = 12.5

    np.testing.assert_allclose(pv.temperature, np.array([12.5, 12.5, 12.5]))


def test_time_stamps_setter_keeps_existing_temperature() -> None:
    """Ensure updating timestamps does not auto-reset temperature when already available."""
    dt = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    ratio = pd.DataFrame(np.ones((len(dt), 1)), index=dt, columns=["area1"])
    pv = PVDisaggregator(pv_ratio=ratio, temperature=np.array([3.0, 4.0]))

    pv.time_stamps = pd.date_range("2024-02-01", periods=2, freq="15min", tz="UTC")

    np.testing.assert_allclose(pv.temperature, np.array([3.0, 4.0]))


@patch("sundael.disaggregation.get_knmi_hourly_temperature_series")
def test_set_temperature_with_custom_range_calls_knmi(mocked_function: MagicMock, knmi_json: str) -> None:
    """Ensure custom date ranges use KNMI path in _set_temperature."""
    mocked_function.return_value = knmi_json
    pv = PVDisaggregator(
        pv_ratio=pd.DataFrame(
            index=pd.DatetimeIndex(pd.date_range("2019-12-31 00:00", "2020-01-01 00:00", freq="15min", tz="UTC"))
        )
    )
    assert pv.temperature.shape == pv.time_stamps.shape


def test_estimate_generation_requires_area_mapping() -> None:
    """Ensure estimate_generation guards against missing area mapping."""
    pv = PVDisaggregator(pv_ratio=pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp("2023-01-01 00:00", tz="UTC")])))
    pv.area_mapping = None
    netload = pd.DataFrame([[0.0]], index=["c1"], columns=[pv.time_stamps[0]])
    params = pd.DataFrame(
        [[0.0, 1.0, 0.65, 0.0, 0.5]], index=["c1"], columns=["base", "effective_size", "tilt", "orientation", "c_temp"]
    )
    with pytest.raises(ValueError, match="Area mapping is None"):
        pv.estimate_generation(netload, params)


def test_validate_area_returns_early_without_mapping() -> None:
    """Ensure _validate_area exits without checks when mapping is None."""
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO)
    pv.area_mapping = None
    pv._validate_area(pd.DataFrame(index=["x"]))


def test_calculate_load_sets_timestamps_when_reference_missing() -> None:
    """Ensure _calculate_load adopts net-load timestamps when no reference is provided."""

    class _LenToggle:
        """Object with controlled length responses for branch testing."""

        def __init__(self) -> None:
            self.calls = 0

        def __len__(self) -> int:
            self.calls += 1
            return 1 if self.calls == 1 else 0

    pv = PVDisaggregator(pv_ratio=None, pv_reference=None)
    # Skip auto temperature update in the time_stamps setter once, then allow
    # _calculate_load to trigger it at line 676.
    pv._temperature = _LenToggle()

    net_load = pd.DataFrame(
        [[1.0, -2.0]],
        index=["c1"],
        columns=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
    )

    with patch.object(PVDisaggregator, "_set_temperature", return_value=np.array([10.0, 11.0])) as set_temp_mock:
        net_con, net_gen, normalized_load = pv._calculate_load(net_con=None, net_gen=None, net_load=net_load)

    assert set_temp_mock.call_count == 1

    assert len(pv.time_stamps) == 2
    pd.testing.assert_index_equal(normalized_load.columns, pv.time_stamps)
    pd.testing.assert_index_equal(net_con.columns, pv.time_stamps)
    pd.testing.assert_index_equal(net_gen.columns, pv.time_stamps)


def test_calculate_load_rejects_timestamp_length_mismatch() -> None:
    """Ensure _calculate_load validates net-load width against reference timestamps."""
    ratio = pd.DataFrame(
        np.ones((2, 1)),
        index=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
        columns=["area1"],
    )
    pv = PVDisaggregator(pv_ratio=ratio, temperature=np.zeros(2))
    net_load = pd.DataFrame(
        [[1.0, -1.0, 0.5]],
        index=["c1"],
        columns=pd.date_range("2024-02-01", periods=3, freq="15min", tz="UTC"),
    )

    with pytest.raises(ValueError, match="same number of measurements"):
        pv._calculate_load(net_con=None, net_gen=None, net_load=net_load)


def test_calculate_load_skips_second_temperature_set_when_already_available() -> None:
    """Ensure _calculate_load does not reset temperature when it is already populated."""
    pv = PVDisaggregator(pv_ratio=None, pv_reference=None)
    net_load = pd.DataFrame(
        [[0.5, -0.5]],
        index=["c1"],
        columns=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
    )

    pv._calculate_load(net_con=None, net_gen=None, net_load=net_load)

    assert len(pv.temperature) == 2


def test_disaggregate_with_sampling_path() -> None:
    """Ensure use_sampling=True runs optimization on filtered time points."""
    dt = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    ratio = pd.DataFrame(np.ones((len(dt), 1)), index=dt, columns=["area1"])
    pv = PVDisaggregator(
        pv_ratio=ratio,
        area_mapping={"c1": "area1"},
        temperature=np.zeros(len(dt)),
    )
    netload = pd.DataFrame([np.linspace(-1, 1, len(dt))], index=["c1"], columns=dt)

    expected_params = pd.DataFrame(
        [[0.0, 1.0, 0.65, 0.0, 0.5]],
        index=["c1"],
        columns=["base", "effective_size", "tilt", "orientation", "c_temp"],
    )
    expected_generation = pd.DataFrame([np.ones(len(dt))], index=["c1"], columns=dt)

    with (
        patch.object(PVDisaggregator, "optimize_parameters", return_value=expected_params) as optimize_mock,
        patch.object(PVDisaggregator, "estimate_generation", return_value=expected_generation),
    ):
        result = pv.disaggregate(net_load=netload, include_self_consumption=False, use_sampling=True)

    assert optimize_mock.call_count == 1
    optimize_arg = optimize_mock.call_args[0][0]
    assert optimize_arg.shape[1] < netload.shape[1]
    assert set(result) == {"params", "net_con", "net_gen", "generation", "consumption"}


def test_disaggregate_with_sampling_cap() -> None:
    """Ensure sampling is capped at N_SAMPLE when enough daytime points exist."""
    dt = pd.date_range("2023-10-01 00:00", "2024-01-15 23:45", freq="15min", tz="UTC")
    ratio = pd.DataFrame(np.ones((len(dt), 1)), index=dt, columns=["area1"])
    pv = PVDisaggregator(
        pv_ratio=ratio,
        area_mapping={"c1": "area1"},
        temperature=np.zeros(len(dt)),
    )

    netload = pd.DataFrame([np.linspace(-1, 1, len(dt))], index=["c1"], columns=dt)

    expected_params = pd.DataFrame(
        [[0.0, 1.0, 0.65, 0.0, 0.5]],
        index=["c1"],
        columns=["base", "effective_size", "tilt", "orientation", "c_temp"],
    )
    expected_generation = pd.DataFrame([np.ones(len(dt))], index=["c1"], columns=dt)

    with (
        patch.object(PVDisaggregator, "optimize_parameters", return_value=expected_params) as optimize_mock,
        patch.object(PVDisaggregator, "estimate_generation", return_value=expected_generation),
    ):
        pv.disaggregate(net_load=netload, include_self_consumption=False, use_sampling=True)

    optimize_arg = optimize_mock.call_args[0][0]
    assert optimize_arg.shape[1] == N_SAMPLE


def test_apply_generation_rules_fallthrough_branch() -> None:
    """Cover the branch where no rule applies due to NaN net consumption."""
    generation = np.array([[5.0]], dtype=np.float64)
    net_gen = np.array([[1.0]], dtype=np.float64)
    net_con = np.array([[np.nan]], dtype=np.float64)
    mean_net_con = np.array([0.0], dtype=np.float64)

    result = apply_generation_rules.py_func(generation, net_gen, net_con, mean_net_con)
    assert result[0, 0] == 5.0


def test_run_estimate_parameters_and_get_solar_properties() -> None:
    """Cover helper wrappers used by multiprocessing code paths."""
    kwargs = {
        "netload": pd.Series([-100.0, 50.0, -25.0]),
        "altitude": np.array([1.0, 0.8, 0.6]),
        "azimuth": np.array([-0.2, 0.0, 0.2]),
        "irradiance": np.array([0.9, 0.85, 0.8]),
        "r_avg": np.array([0.95, 0.96, 0.94]),
        "temperature": DEFAULT_CONFIG.yearly_avg_temp,
    }
    params = run_estimate_parameters(kwargs)
    assert params.shape == (5,)

    solar = get_solar_properties(PVDisaggregator(pv_ratio=EMPTY_RATIO).area_solar_values("default"))
    assert len(solar) == 4


@pytest.mark.parametrize("year", (2020, 2021, 2022, 2023, 2024))
@patch("sundael.temperature.get_knmi_hourly_temperature_series")
def test_disaggregation_temperature_year(mocked_function: MagicMock, year: int) -> None:
    """Test temperature when year is supplied to PVDisaggregator.

    Ensure KNMI API is not called.
    """
    pv = PVDisaggregator(
        pv_ratio=pd.DataFrame(index=pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:45", freq="15min", tz="UTC"))
    )
    assert pv.temperature.shape[0] in (35040, 35136)


@pytest.fixture
def knmi_json() -> str:
    """Mocked KNMI API json output."""
    with open("tests/data/temperature/knmi_output.json") as f:
        json = f.read()

    return json


@patch("sundael.temperature.get_knmi_hourly_temperature_series")
def test_disaggregation_temperature_knmi(mocked_function: MagicMock, knmi_json: str) -> None:
    """Test temperature from KNMI in PVDisaggregator."""
    mocked_function.return_value = knmi_json
    pv = PVDisaggregator(
        pv_ratio=pd.DataFrame(index=pd.date_range("2023-01-01 00:00", "2023-12-31 23:45", freq="15min", tz="UTC"))
    )
    assert pv.temperature.shape == (35040,)


@patch("sundael.temperature.get_knmi_hourly_temperature_series")
def test_disaggregation_temperature_knmi_date_range(mocked_function: MagicMock, knmi_json: str) -> None:
    """Test temperature from KNMI in PVDisaggregator."""
    mocked_function.return_value = knmi_json
    pv = PVDisaggregator(
        pv_ratio=pd.DataFrame(index=pd.date_range("2019-01-01", "2019-12-31 23:45", freq="15min", tz="UTC"))
    )
    assert pv.temperature.shape == (35040,)


@pytest.fixture
def disagg_data_dir() -> Path:
    """Disaggregation test data directory."""
    return Path("tests/data/disaggregation/")


@pytest.fixture
def full_net_con(disagg_data_dir: Path) -> pd.DataFrame:
    """Disaggregation test net consumption data."""
    return pd.read_parquet(disagg_data_dir / "net_con.parquet").T


@pytest.fixture
def full_net_gen(disagg_data_dir: Path) -> pd.DataFrame:
    """Disaggregation test net generation data."""
    return pd.read_parquet(disagg_data_dir / "net_gen.parquet").T


@pytest.fixture
def full_netload(full_net_con: pd.DataFrame, full_net_gen: pd.DataFrame) -> pd.DataFrame:
    """Disaggregation test netload data."""
    return full_net_gen - full_net_con


@pytest.fixture
def full_pv_ratio(disagg_data_dir: Path) -> pd.DataFrame:
    """Disaggregation test pv_ratio data."""
    return pd.read_parquet(disagg_data_dir / "pv_ratio.parquet").T


@pytest.mark.parametrize(
    "net_con,net_gen,netload,pv_ratio",
    (("full_net_con", "full_net_gen", None, "full_pv_ratio"), (None, None, "full_netload", "full_pv_ratio")),
)
def test_disaggregation_e2e_without_ratio(
    net_con: str | None, net_gen: str | None, netload: str | None, pv_ratio: str, request: pytest.FixtureRequest
) -> None:
    """Test a full disaggregation example."""
    net_con = request.getfixturevalue(net_con) if net_con is not None else None
    net_gen = request.getfixturevalue(net_gen) if net_gen is not None else None
    netload = request.getfixturevalue(netload) if netload is not None else None
    pv_ratio = request.getfixturevalue(pv_ratio).T if pv_ratio is not None else pd.DataFrame()

    pv = PVDisaggregator(pv_ratio=pv_ratio)
    result = pv.disaggregate(net_con=net_con, net_gen=net_gen, net_load=netload, include_self_consumption=True)

    assert result["consumption"].min().min() == 0
    assert result["generation"].min().min() == 0

    np.testing.assert_almost_equal(result["consumption"].max().max(), 12.0)
    np.testing.assert_almost_equal(result["generation"].max().max(), 20.86, decimal=2)

    # Example of a disaggregated timepoint
    assert result["consumption"].iloc[0, 15003] > 3.9
    assert result["generation"].iloc[0, 15003] > 3.9


def test_disaggregation_e2e_with_ratio(full_netload: pd.DataFrame, full_pv_ratio: pd.DataFrame) -> None:
    """Test a full disaggregation example without applying fixes."""
    pv = PVDisaggregator(pv_ratio=full_pv_ratio.T)
    result = pv.disaggregate(net_load=full_netload, include_self_consumption=False)

    # Example of a disaggregated timepoint
    assert result["consumption"].iloc[0, 15004] > 4
    assert result["generation"].iloc[0, 15004] > 4


def test_disaggregation_mpi(full_netload: pd.DataFrame, full_pv_ratio: pd.DataFrame) -> None:
    """Test a full disaggregation example and check the mpi."""
    pv = PVDisaggregator(pv_ratio=full_pv_ratio.T)
    pv.disaggregate(net_load=full_netload, include_self_consumption=True)

    pd.testing.assert_frame_equal(
        pv.mpi,
        pd.DataFrame(
            {
                "variable": ["example1"],
                "year": [2024],
                "self_consumption_mpi": [1.001522],
                "mean_self_consumption": [0.350545],
                "max_self_consumption": [7.671745],
            },
        ),
        check_dtype=False,
        atol=0.001,
    )


def test_disaggregation_not_net_con_net_gen_and_netloadtest_disaggregation_mpi_supplied(
    full_net_con: pd.DataFrame,
) -> None:
    """Supplying net_con, net_gen AND netload should give an error."""
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO)
    with pytest.raises(ValueError):
        pv.disaggregate(net_con=full_net_con, net_gen=full_net_con, net_load=full_net_con)


def test_disaggregation_net_con_net_gen_and_netload_not_supplied() -> None:
    """Supplying none o fnet_con, net_gen and netload should give an error."""
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO)
    with pytest.raises(ValueError):
        pv.disaggregate(net_con=None, net_gen=None, net_load=None)


@pytest.mark.parametrize(
    "pc,lat,lon",
    (
        ("7223", 52.07, 6.23),
        ("1011", 52.37, 4.91),
        ("default", DEFAULT_CONFIG.latitude, DEFAULT_CONFIG.longitude),
    ),
)
def test_lat_lon(pc: str, lat: float, lon: float) -> None:
    """Test obtaining solar values based on latitude and longitude."""
    latlon = pd.read_table("tests/data/disaggregation/area_latlon.txt", index_col="pc")
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO, area_latlon=latlon)
    np.testing.assert_almost_equal(pv.area_solar_values(pc).latitude, lat)
    np.testing.assert_almost_equal(pv.area_solar_values(pc).longitude, lon)


@pytest.mark.parametrize(
    "pc,lat,lon",
    (
        ("7223", DEFAULT_CONFIG.latitude, DEFAULT_CONFIG.longitude),
        ("1011", DEFAULT_CONFIG.latitude, DEFAULT_CONFIG.longitude),
        ("default", DEFAULT_CONFIG.latitude, DEFAULT_CONFIG.longitude),
    ),
)
def test_lat_lon_default(pc: str, lat: float, lon: float) -> None:
    """Test obtaining solar values based on default latitude and longitude."""
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO)
    np.testing.assert_almost_equal(pv.area_solar_values(pc).latitude, lat)
    np.testing.assert_almost_equal(pv.area_solar_values(pc).longitude, lon)


def test_correct_generation_with_net_gen() -> None:
    """Test correction of disaggregated generation."""
    netload = pd.DataFrame(
        [
            [10, 10, 20, 30, 20, 10, -10],  # max 30
            [10, 10, 20, 30, 20, 10, -10],  # max 30
            [-100, -100, 10, 20, 30, 20, 10],  # max 30
            [0, 1, 0, 10, 0, 0, 1],
        ],  # max 10
        dtype=np.float64,
    )
    net_gen = netload.copy()
    net_gen[net_gen < 0] = 0

    generation = pd.DataFrame(
        [
            [30, 40, 40, 40, 40, 40, 30],  # max 40
            [10, 15, 25, 25, 25, 10, 0],  # max 25
            [0, 0, 0, 0, 0, 0, 0],  # max 0
            [0, 1, 0, 1, 0, 0, 1],
        ],  # max 1
        dtype=np.float64,
    )

    expected = pd.DataFrame(
        [
            [30.0, 40.0, 40.0, 40.0, 40.0, 40.0, 30.0],
            np.array([10.0, 15.0, 25.0, 25.0, 25.0, 10.0, 0.0]) * 1.2,
            [0.0, 0.0, 10.0, 20.0, 30.0, 20.0, 10.0],
            [0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    result = correct_generation_with_net_gen(generation=generation, net_gen=net_gen)
    pd.testing.assert_frame_equal(result, expected, check_names=False)


RuleBasedTestData = namedtuple("RuleBasedTestData", "net_con net_gen generation expected")


@pytest.fixture
def rule_based_data() -> RuleBasedTestData:
    """Test data for rule-based self consumption tests."""
    data = RuleBasedTestData(
        np.array(
            [
                [1, 10, 0, 0],
                [0, 0, 0, 4],
                [35, 2, 10, 0],
                [35, 2, 10, 0],
                [15, 1, 4, 0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [1, 50, 0, 0],
                [35, 2, 10, 0],
                [0, 0, 0, 0],
                [15, 1, 5, 0],
                [15, 2, 5, 0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0, 40, 400, 0],
                [35, 2, 35, 0],
                [35, 2, 10, 0],
                [15, 2, 10, 0],
                [55, 10, 6, 0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [35, 2.0, 10.0, 0.0],
                [0.0, 1.0, 5.0, 0.0],
                [15.0, 1.0, 1.0, 0.0],
            ]
        ),
    )

    return data


def test_calculate_rule_based_self_consumption(rule_based_data: RuleBasedTestData) -> None:
    """Test rule-based calculation of consumption."""
    net_con = pd.DataFrame(rule_based_data.net_con)
    net_gen = pd.DataFrame(rule_based_data.net_gen)
    generation = pd.DataFrame(rule_based_data.generation)
    expected = pd.DataFrame(rule_based_data.expected)

    result = calculate_rule_based_self_consumption(generation=generation, net_con=net_con, net_gen=net_gen)
    pd.testing.assert_frame_equal(result, expected, check_names=False)


def test_apply_generation_rules(rule_based_data: RuleBasedTestData) -> None:
    """Test numba implementation of rule-based calculation of consumption."""
    mean_net_con = np.nan_to_num(mean_gt_zero(rule_based_data.net_con, axis=1))
    result = apply_generation_rules.py_func(
        rule_based_data.generation, rule_based_data.net_gen, rule_based_data.net_con, mean_net_con
    )

    np.testing.assert_array_almost_equal(result, rule_based_data.expected)


SimpleTestData = namedtuple("SimpleTestData", "net_gen generation expected mpi")


@pytest.fixture
def simple_test_data() -> SimpleTestData:
    """Test data for simple self consumption calculation."""
    columns = pd.date_range(start="2023-01-01 12:00", freq="1D", periods=4)
    data = SimpleTestData(
        pd.DataFrame(
            [
                [1, 50, 0, 0],
                [35, 2, 10, 0],
                [0, 0, 0, 0],
                [15, 1, 5, 0],
                [15, 2, 5, 0],
            ],
            columns=columns,
            dtype=np.float64,
        ),
        pd.DataFrame(
            [
                [0, 40, 400, 0],
                [35, 2, 35, 0],
                [35, 2, 10, 0],
                [15, 2, 10, 0],
                [55, 10, 6, 0],
            ],
            columns=columns,
            dtype=np.float64,
        ),
        pd.DataFrame(
            [
                [0.0, 0.0, 400.0, 0.0],
                [0.0, 0.0, 25.0, 0.0],
                [35.0, 2.0, 10.0, 0.0],
                [0.0, 1.0, 5.0, 0.0],
                [40.0, 8.0, 1.0, 0.0],
            ],
            columns=columns,
        ),
        pd.DataFrame(
            {
                "variable": ["0", "1", "2", "3", "4"],
                "year": [2023] * 5,
                "self_consumption_mpi": [1.025, 1, 1, 1, 1],
            }
        ),
    )
    return data


def test_calculate_simple_self_consumption(simple_test_data: SimpleTestData) -> None:
    """Test correction of disaggregated generation."""
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO)
    result = pv.calculate_simple_self_consumption(
        generation=simple_test_data.generation, net_gen=simple_test_data.net_gen
    )
    pd.testing.assert_frame_equal(result, simple_test_data.expected, check_names=False)


def test_calculate_simple_self_consumption_mpi(simple_test_data: SimpleTestData) -> None:
    """Test correction of disaggregated generation."""
    pv = PVDisaggregator(pv_ratio=EMPTY_RATIO)
    pv.calculate_simple_self_consumption(generation=simple_test_data.generation, net_gen=simple_test_data.net_gen)
    pd.testing.assert_frame_equal(pv.mpi, simple_test_data.mpi, check_dtype=False)


temp = [10] * (24 * 4 + 1)


@pytest.mark.parametrize("temperature", (pd.Series(temp), pl.Series(temp)))
def test_pvdisaggregator_with_temperature(temperature: pl.Series | pd.Series) -> None:
    """Test temperature can be set with polars or pandas series."""
    PVDisaggregator(
        pv_ratio=pd.DataFrame(index=pd.date_range("2023-01-01", periods=len(temp), freq="15min", tz="UTC")),
        temperature=temperature,
    )


def test_pvdisaggregator_with_temperature_error_in_temperature_shape() -> None:
    """Ensure temperature shape is checked."""
    temperature = np.array([10, 12, 13])
    with pytest.raises(ValueError):
        PVDisaggregator(
            pv_ratio=pd.DataFrame(index=pd.date_range("2023-01-01", periods=10, freq="15min", tz="UTC")),
            temperature=temperature,
        )


@pytest.mark.parametrize(
    ("netload", "area_mapping", "error"),
    [
        (pd.DataFrame(index=["bla", "flop"]), {"bla": "1011", "flop": "1011"}, None),
        (pd.DataFrame(index=["bla", "flop"]), {"bla": "1011", "flop": "1012"}, ValueError),
        (pd.DataFrame(index=["bla", "flop"]), {"bla": "1013", "flop": "1014"}, ValueError),
    ],
)
def test_validate_area(
    netload: pd.DataFrame, area_mapping: dict[str, str], error: type[Exception] | None, full_pv_ratio: pd.DataFrame
) -> None:
    """Test that a ValueError is raised when pc4 of netload is not in pv_ratio."""
    with pytest.raises(error, match="not present") if error is not None else contextlib.nullcontext():
        pv = PVDisaggregator(pv_ratio=full_pv_ratio.T, area_mapping=area_mapping)
        pv._validate_area(netload)
