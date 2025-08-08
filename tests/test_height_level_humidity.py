import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry

from .utils import SelectAndAddFieldSource
from .utils import assert_fields_equal
from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20.0, 40.0, 60.0, 80.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

R2M_VALUES = np.array([[0, 10, 20, 30], [40, 50, 60, 70], [80, 90, 100, 110]])
T2M_VALUES = np.array([[299, 295, 294, 291], [286, 269, 291, 291], [297, 299, 250, 238]])
Q2M_VALUES = np.array(
    [
        [0.0, 0.0020382843983213, 0.0030328053695092, 0.0041460924390494],
        [0.0037995906399138, 0.0013875242433219, 0.007505377077515, 0.0087464703870742],
        [0.0146151303517757, 0.0191325953775226, 0.0004603447218772, 0.0001570271615703],
    ]
)
D2M_VALUES = np.array(
    [
        [170.15730777, 262.02113729, 270.2148745, 273.19127971],
        [272.73039562, 259.95112967, 283.13080014, 285.45374602],
        [293.33820197, 297.23259971, 247.36689612, 235.17086978],
    ]
)
SP_VALUES = np.array([[80417, 79975, 101152, 92153], [97221, 99400, 102021, 102212], [101488, 98555, 101390, 91533]])

T_VALUES = {
    136: np.array(
        [[297.67, 295.83, 295.22, 292.19], [288.51, 269.26, 290.87, 291.02], [296.24, 298.50, 250.34, 244.79]]
    ),
    137: np.array(
        [[297.82, 294.65, 294.48, 291.79], [287.30, 269.31, 291.06, 291.21], [296.50, 298.53, 250.34, 243.16]]
    ),
}

Q_VALUES = {
    136: np.array(
        [
            [0.000967, 0.001873, 0.002905, 0.004181],
            [0.003655, 0.001024, 0.006701, 0.008176],
            [0.013856, 0.018466, 0.000482, 0.000268],
        ]
    ),
    137: np.array(
        [
            [0.000991, 0.001986, 0.002948, 0.004209],
            [0.003709, 0.001124, 0.006963, 0.008426],
            [0.013991, 0.018540, 0.000485, 0.000245],
        ]
    ),
}

# IFS A and B coeffients for level 137 - 129
AB_coefficients_files = {
    "A": [424.414063, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813, 0.0, 0.0],
    "B": [0.969513, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.000000],
}

# IFS A and B coeffients for level 137 - 129
AB_coefficients = {
    "A": [3.757813, 0.0, 0.0],
    "B": [0.995003, 0.997630, 1.000000],
}


@pytest.fixture
def relative_humidity_source(test_source):
    HEIGHT_LEVEL_RELATIVE_HUMIDITY_SPEC = [
        {"param": "2r", "values": R2M_VALUES, **MOCK_FIELD_METADATA},
        {"param": "sp", "values": SP_VALUES, **MOCK_FIELD_METADATA},
        {"param": "2t", "values": T2M_VALUES, **MOCK_FIELD_METADATA},
    ]
    for level, values in T_VALUES.items():
        HEIGHT_LEVEL_RELATIVE_HUMIDITY_SPEC.append(
            {"param": "t", "levtype": "ml", "levelist": level, "values": values, **MOCK_FIELD_METADATA}
        )
    for level, values in Q_VALUES.items():
        HEIGHT_LEVEL_RELATIVE_HUMIDITY_SPEC.append(
            {"param": "q", "levtype": "ml", "levelist": level, "values": values, **MOCK_FIELD_METADATA}
        )
    return test_source(HEIGHT_LEVEL_RELATIVE_HUMIDITY_SPEC)


@pytest.fixture
def specific_humidity_source(test_source):
    HEIGHT_LEVEL_SPECIFIC_HUMIDITY_SPEC = [
        {"param": "2sh", "values": Q2M_VALUES, **MOCK_FIELD_METADATA},
        {"param": "sp", "values": SP_VALUES, **MOCK_FIELD_METADATA},
        {"param": "2t", "values": T2M_VALUES, **MOCK_FIELD_METADATA},
    ]
    for level, values in T_VALUES.items():
        HEIGHT_LEVEL_SPECIFIC_HUMIDITY_SPEC.append(
            {"param": "t", "levtype": "ml", "levelist": level, "values": values, **MOCK_FIELD_METADATA}
        )
    for level, values in Q_VALUES.items():
        HEIGHT_LEVEL_SPECIFIC_HUMIDITY_SPEC.append(
            {"param": "q", "levtype": "ml", "levelist": level, "values": values, **MOCK_FIELD_METADATA}
        )
    return test_source(HEIGHT_LEVEL_SPECIFIC_HUMIDITY_SPEC)


@pytest.fixture
def dewpoint_temperature_source(test_source):
    HEIGHT_LEVEL_DEWPOINT_TEMPERATURE_SPEC = [
        {"param": "2d", "values": D2M_VALUES, **MOCK_FIELD_METADATA},
        {"param": "sp", "values": SP_VALUES, **MOCK_FIELD_METADATA},
    ]
    for level, values in T_VALUES.items():
        HEIGHT_LEVEL_DEWPOINT_TEMPERATURE_SPEC.append(
            {"param": "t", "levtype": "ml", "levelist": level, "values": values, **MOCK_FIELD_METADATA}
        )
    for level, values in Q_VALUES.items():
        HEIGHT_LEVEL_DEWPOINT_TEMPERATURE_SPEC.append(
            {"param": "q", "levtype": "ml", "levelist": level, "values": values, **MOCK_FIELD_METADATA}
        )
    return test_source(HEIGHT_LEVEL_DEWPOINT_TEMPERATURE_SPEC)


@skip_if_offline
def test_height_level_specific_humidity_to_relative_humidity_from_file(test_source):
    source = test_source("anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib")
    q_to_r_height = filter_registry.create(
        "q_to_r_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients_files,
    )

    pipeline = source | q_to_r_height

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2d", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2d", "2t", "sp", "2r"}

    # test unchanged fields agree
    for param in ("2sh", "2d", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2r"][0].to_numpy()
    expected_relative_humidity = test_source(
        "anemoi-transform/filters/single_level_relative_humidity.npy"
    ).ds.to_numpy()
    np.testing.assert_allclose(result, expected_relative_humidity)


def test_specific_humidity_to_relative_humidity(specific_humidity_source):
    q_to_r_height = filter_registry.create(
        "q_to_r_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )
    pipeline = specific_humidity_source | q_to_r_height

    input_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2t", "sp", "2r"}

    # test unchanged fields agree
    for param in ("2sh", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2r"][0].to_numpy()
    expected_relative_humidity = R2M_VALUES
    np.testing.assert_allclose(result, expected_relative_humidity)


def test_height_level_specific_humidity_to_relative_humidity_round_trip(specific_humidity_source):
    q_to_r_height = filter_registry.create(
        "q_to_r_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    r_to_q_height = filter_registry.create(
        "r_to_q_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    relative_humidity_source = SelectAndAddFieldSource(
        specific_humidity_source | q_to_r_height,
        specific_humidity_source,
        params=["2r", "2t", "sp"],
        additional_params=["q", "t"],
    )

    pipeline = relative_humidity_source | r_to_q_height

    input_fields = collect_fields_by_param(specific_humidity_source)
    intermediate_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2t", "sp", "t", "q"}
    assert set(intermediate_fields) == {"2r", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2r", "2t", "sp"}

    # test unchanged fields agree from beginning to end
    for param in ("2sh", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("2r", "2t", "sp"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


@skip_if_offline
def test_relative_humidity_to_specific_humidity_from_file(test_source):
    source = test_source("anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib")
    input_relative_humidity = test_source("anemoi-transform/filters/single_level_relative_humidity.npy").ds.to_numpy()

    md = source.ds.sel(param="2d")[0].metadata().override(edition=2, shortName="2r")

    source.ds += ekd.FieldList.from_array(input_relative_humidity, md)

    r_to_q_height = filter_registry.create(
        "r_to_q_height",
        height=2,
        specific_humidity_at_height_level="2q",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients_files,
    )

    pipeline = source | r_to_q_height

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2r", "2d", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2r", "2d", "2t", "sp", "2q"}

    # test unchanged fields agree
    for param in ("2sh", "2r", "2d", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2q"][0].to_numpy()
    expected_specific_humidity = output_fields["2sh"][0].to_numpy()
    np.testing.assert_allclose(result, expected_specific_humidity)


def test_relative_humidity_to_specific_humidity(relative_humidity_source):
    r_to_q_height = filter_registry.create(
        "r_to_q_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )
    pipeline = relative_humidity_source | r_to_q_height

    input_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2r", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2r", "2t", "sp", "2sh"}

    # test unchanged fields agree
    for param in ("2r", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2sh"][0].to_numpy()
    expected_specific_humidity = Q2M_VALUES
    np.testing.assert_allclose(result, expected_specific_humidity)


def test_height_level_relative_humidity_to_specific_humidity_round_trip(relative_humidity_source):
    r_to_q_height = filter_registry.create(
        "r_to_q_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    q_to_r_height = filter_registry.create(
        "q_to_r_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    specific_humidity_source = SelectAndAddFieldSource(
        relative_humidity_source | r_to_q_height,
        relative_humidity_source,
        params=["2sh", "2t", "sp"],
        additional_params=["q", "t"],
    )

    pipeline = specific_humidity_source | q_to_r_height

    input_fields = collect_fields_by_param(relative_humidity_source)
    intermediate_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2r", "2t", "sp", "t", "q"}
    assert set(intermediate_fields) == {"2sh", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2r", "2t", "sp"}

    # test unchanged fields agree from beginning to end
    for param in ("2r", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("2sh", "2t", "sp"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


@skip_if_offline
def test_specific_humidity_to_dewpoint_from_file(test_source):
    source = test_source("anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib")

    q_to_d_height = filter_registry.create(
        "q_to_d_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        dewpoint_temperature_at_height_level="2td",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients_files,
    )

    pipeline = source | q_to_d_height

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2d", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2d", "2t", "sp", "2td"}

    # test unchanged fields agree
    for param in ("2sh", "2d", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2td"][0].to_numpy()
    expected_dewpoint_temperature = test_source(
        "anemoi-transform/filters/single_level_dewpoint_temperature.npy"
    ).ds.to_numpy()
    np.testing.assert_allclose(result, expected_dewpoint_temperature)


def test_specific_humidity_to_dewpoint_temperature(specific_humidity_source):
    q_to_d_height = filter_registry.create(
        "q_to_d_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        dewpoint_temperature_at_height_level="2d",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )
    pipeline = specific_humidity_source | q_to_d_height

    input_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2t", "sp", "2d"}

    # test unchanged fields agree
    for param in ("2sh", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2d"][0].to_numpy()
    expected_relative_humidity = D2M_VALUES
    np.testing.assert_allclose(result, expected_relative_humidity)


def test_height_level_specific_humidity_to_dewpoint_temperature_round_trip(specific_humidity_source):
    q_to_d_height = filter_registry.create(
        "q_to_d_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        dewpoint_temperature_at_height_level="2d",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    d_to_q_height = filter_registry.create(
        "d_to_q_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        dewpoint_temperature_at_height_level="2d",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    dewpoint_temperature_source = SelectAndAddFieldSource(
        specific_humidity_source | q_to_d_height,
        specific_humidity_source,
        params=["2d", "sp"],
        additional_params=["2t", "q", "t"],
    )

    pipeline = dewpoint_temperature_source | d_to_q_height

    input_fields = collect_fields_by_param(specific_humidity_source)
    intermediate_fields = collect_fields_by_param(dewpoint_temperature_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2t", "sp", "t", "q"}
    assert set(intermediate_fields) == {"2d", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2t", "2d", "sp"}

    # test unchanged fields agree from beginning to end
    for param in ("2sh", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("2d", "2t", "sp"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


@skip_if_offline
def test_dewpoint_to_specific_humidity_from_file(test_source):
    source = test_source("anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib")
    input_dewpoint_temperature = test_source(
        "anemoi-transform/filters/single_level_dewpoint_temperature.npy"
    ).ds.to_numpy()
    md = source.ds.sel(param="2d")[0].metadata()
    ds = source.ds.sel(param=["2sh", "2t", "sp", "q", "t"])
    ds += ekd.FieldList.from_array(input_dewpoint_temperature, md)
    source.ds = ds

    d_to_q_height = filter_registry.create(
        "d_to_q_height",
        height=2,
        specific_humidity_at_height_level="2q",
        dewpoint_temperature_at_height_level="2d",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients_files,
    )

    pipeline = source | d_to_q_height

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2d", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2d", "2t", "sp", "2q"}

    # test unchanged fields agree
    for param in ("2sh", "2d", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2q"][0].to_numpy()
    expected_specific_humidity = output_fields["2sh"][0].to_numpy()
    np.testing.assert_allclose(result, expected_specific_humidity)


def test_dewpoint_temperature_to_specific_humidity(dewpoint_temperature_source):
    d_to_q_height = filter_registry.create(
        "d_to_q_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        dewpoint_temperature_at_height_level="2d",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )
    pipeline = dewpoint_temperature_source | d_to_q_height

    input_fields = collect_fields_by_param(dewpoint_temperature_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2d", "sp", "t", "q"}
    assert set(output_fields) == {"2d", "sp", "2sh"}

    # test unchanged fields agree
    for param in ("2d", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    result = output_fields["2sh"][0].to_numpy()
    expected_specific_humidity = Q2M_VALUES
    np.testing.assert_allclose(result, expected_specific_humidity)


def test_height_level_dewpoint_temperature_to_specific_humidity_round_trip(dewpoint_temperature_source):
    d_to_q_height = filter_registry.create(
        "d_to_q_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        dewpoint_temperature_at_height_level="2d",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    q_to_d_height = filter_registry.create(
        "q_to_d_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        dewpoint_temperature_at_height_level="2d",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        model_level_AB=AB_coefficients,
    )

    specific_humidity_source = SelectAndAddFieldSource(
        dewpoint_temperature_source | d_to_q_height,
        dewpoint_temperature_source,
        params=["2sh", "sp"],
        additional_params=["q", "t"],
    )

    pipeline = specific_humidity_source | q_to_d_height

    input_fields = collect_fields_by_param(dewpoint_temperature_source)
    intermediate_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2d", "sp", "t", "q"}
    assert set(intermediate_fields) == {"2sh", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2d", "sp"}

    # test unchanged fields agree from beginning to end
    for param in ("2d", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("2sh", "sp"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
