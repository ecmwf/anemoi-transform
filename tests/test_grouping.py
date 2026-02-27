import earthkit.data as ekd
import pytest

from anemoi.transform.grouping import GroupByParam

from .utils import mock_field


def field_generator(**metadata_values):
    MOCK_MARS_METADATA = {
        "domain": "g",
        "levtype": "sfc",
        "date": 20200513,
        "time": 1200,
        "step": 0,
        "param": "2t",
        "class": "od",
        "type": "an",
        "stream": "oper",
        "expver": "0001",
    }
    # builds fields with metadata from cartesian product of metadata_values
    fields = []
    import itertools

    combinations = itertools.product(*metadata_values.values())
    for values in combinations:
        metadata = MOCK_MARS_METADATA | dict(zip(metadata_values.keys(), values))
        fields.append(mock_field(**metadata))
    return fields


@pytest.fixture
def sample_fields():
    return field_generator(
        step=[0, 1],
        param=["t", "q", "u", "v"],
    )


@pytest.fixture
def sample_fields_vertical():
    surface_fields = field_generator(step=[0, 1], levtype=["sfc"], param=["2q", "2r", "2t", "sp"])
    vertical_fields = field_generator(step=[0, 1], levtype=["ml"], param=["q", "t"], levelist=[1, 2, 3])
    return surface_fields + vertical_fields


def test_group_by_param(sample_fields):
    match_params = ["u", "v"]
    grouper = GroupByParam(params=match_params)

    # capture non-matching fields in other
    other = []
    num_matching = 0
    for group in grouper.iterate(sample_fields, other=other.append):
        assert len(group) == len(match_params)
        # ensure order is the same
        assert [field.metadata("param") for field in group] == match_params
        metadata = []
        for field in group:
            num_matching += 1
            # check field is unchanged
            assert field in sample_fields

            # get metadata except param from each field
            m = field.metadata(namespace="mars")
            m.pop("param", None)
            metadata.append(m)
        # rest of the metadata the same within a group
        assert all(m == metadata[0] for m in metadata[1:])

    assert num_matching + len(other) == len(sample_fields)
    for field in other:
        assert field.metadata("param") not in match_params
        assert field in sample_fields


@pytest.mark.xfail(reason="vertical grouping not yet implemented")
def test_group_by_param_vertical(sample_fields_vertical):
    from anemoi.transform.grouping import GroupByParamVertical

    def get_param(f):
        if isinstance(f, ekd.Field):
            f = [f]

        param = [x.metadata("param") for x in f]
        assert len(set(param)) == 1
        return param[0]

    match_params = ["sp", "q", "t"]
    grouper = GroupByParamVertical(params=match_params)

    # capture non-matching fields in other
    other = []
    num_matching = 0
    for group in grouper.iterate(sample_fields_vertical, other=other.append):
        assert len(group) == len(match_params)
        # ensure order is the same
        assert [get_param(field) for field in group] == match_params
        metadata = []
        for fields in group:
            if isinstance(fields, ekd.Field):
                fields = [fields]
            for field in fields:
                num_matching += 1
                # check field is unchanged
                assert field in sample_fields_vertical
                # get metadata except keys known to be different from each field
                m = field.metadata(namespace="mars")
                m.pop("param", None)
                m.pop("levtype", None)
                m.pop("levelist", None)
                metadata.append(m)

        # rest of the metadata the same within a group
        assert all(m == metadata[0] for m in metadata[1:])

    assert num_matching + len(other) == len(sample_fields_vertical)
    for field in other:
        assert field.metadata("param") not in match_params
        assert field in sample_fields_vertical
