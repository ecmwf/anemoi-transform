import datetime

import numpy as np
import pytest

from anemoi.transform import Field
from anemoi.transform import FieldList


@pytest.fixture
def fieldlist():
    return FieldList.from_source("sample", "test.grib")


@pytest.fixture()
def field(fieldlist):
    return fieldlist[0]


@pytest.fixture()
def field_step_6():
    return FieldList.from_source("sample", "pl.grib").sel(**{"time.step": datetime.timedelta(hours=6)})[0]


def test_new_fieldlist_from_list(fieldlist):
    fields = list(fieldlist)
    result = FieldList.from_fields(fields)
    assert isinstance(result, FieldList)
    assert len(result) == len(fields)
    # ensure using the same objects (not copies)
    assert all(id(f) == id(r) for f, r in zip(fields, result))


def test_new_empty_fieldlist():
    result = FieldList()
    assert isinstance(result, FieldList)
    assert len(result) == 0


def test_new_field_from_numpy_data_only(field):
    array = field.to_numpy() + 1

    result = Field.from_numpy(array, template=field)
    assert isinstance(result, Field)

    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), array)


def test_new_field_from_numpy_update_param(field):
    array = field.to_numpy() + 1

    result = Field.from_numpy(array, template=field, param="foo")
    assert isinstance(result, Field)

    assert result.parameter.variable() != field.parameter.variable()
    assert result.parameter.variable() == "foo"

    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), array)


def test_new_field_from_numpy_update_number(field):
    array = field.to_numpy() + 1

    result = Field.from_numpy(array, template=field, number=99)
    assert isinstance(result, Field)

    assert field.ensemble.member() != result.ensemble.member()
    # ensemble.member() returns a str
    assert result.ensemble.member() == "99"

    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), array)


def test_new_field_from_numpy_update_levelist(field):
    array = field.to_numpy() + 1

    result = Field.from_numpy(array, template=field, levelist=99)
    assert isinstance(result, Field)

    assert field.vertical.level() == 0
    assert result.vertical.level() == 99

    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), array)


def test_new_field_with_valid_datetime(field_step_6):
    field = field_step_6
    assert field.time.step() == datetime.timedelta(hours=6)

    new_valid_datetime = field.time.valid_datetime() - field.time.step()
    assert new_valid_datetime == field.time.base_datetime()

    result = Field.with_valid_datetime(field, new_valid_datetime)
    assert isinstance(result, Field)

    # check valid_datetime and step are updated (step set to 0) - base datetime unchanged
    # ie. valid_datetime is the same as base_datetime
    assert result.time.valid_datetime() == new_valid_datetime
    assert result.time.base_datetime() == field.time.base_datetime()
    assert result.time.step() != field.time.step()
    assert result.time.step() == datetime.timedelta(hours=0)

    # check data unchanged
    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), field.to_numpy())


def test_new_field_with_metadata_update_param(field):
    # new_field_with_metadata works similar to new_field_from_numpy except
    # it does not allow for updating the data
    result = Field.with_new_metadata(field, param="foo")
    assert isinstance(result, Field)

    # check param updated
    assert result.parameter.variable() == "foo"
    assert result.parameter.variable() != field.parameter.variable()

    # check data unchanged
    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), field.to_numpy())


def test_new_field_with_metadata_update_param_and_levelist(field):
    # new_field_with_metadata works similar to new_field_from_numpy except
    # it does not allow for updating the data
    result = Field.with_new_metadata(field, param="foo", levelist=99)
    assert isinstance(result, Field)

    # check param and level updated
    assert result.parameter.variable() == "foo"
    assert result.parameter.variable() != field.parameter.variable()
    assert field.vertical.level() == 0
    assert result.vertical.level() == 99

    # check data unchanged
    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), field.to_numpy())


def test_new_field_from_latitudes_longitudes(field):
    lat, lon = field.geography.latlons()
    new_lat = lat + 12
    new_lon = lon - 34

    result = Field.from_latitudes_longitudes(field, new_lat, new_lon)
    assert isinstance(result, Field)

    # check grid points updated
    result_lat, result_lon = result.geography.latlons()
    assert np.array_equal(result_lat, new_lat)
    assert np.array_equal(result_lon, new_lon)

    # check data unchanged
    assert result.shape == field.shape
    assert np.array_equal(result.to_numpy(), field.to_numpy())
