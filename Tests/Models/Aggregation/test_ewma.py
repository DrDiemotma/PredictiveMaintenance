import pytest
from Models.Aggregation import EwmaTest, EwmaDirection, EwmaResult


@pytest.mark.parametrize("direction,values,threshold,expected_exceedances", [
    (EwmaDirection.BOTH, [0, 1, 2, 3], 2.0, [False, False, False, True]),
    (EwmaDirection.UPPER_BOUNDARY, [0, 1, 2, 3], 2.0, [False, False, False, True]),
    (EwmaDirection.LOWER_BOUNDARY, [0, -1, -2, -3], 2.0, [False, False, False, True]),
])
def test_ewma_basic(direction, values, threshold, expected_exceedances):
    alpha = 0.5
    ewma = EwmaTest(initial_value=0.0, alpha=alpha, threshold=threshold, direction=direction)

    results = [ewma.next(v) for v in values]

    for res, expected in zip(results, expected_exceedances):
        assert res.exceeds_threshold == expected, f"Value: {res.filtered_value}, Expected: {expected}"


def test_ewma_reset():
    ewma = EwmaTest(initial_value=0.0, alpha=0.5, threshold=1.0)
    ewma.next(5.0)
    ewma.reset(0.0)
    result = ewma.next(0.0)
    assert abs(result.filtered_value) < 1e-8, "EWMA should be reset to 0"
    assert not result.exceeds_threshold, "Threshold should not be exceeded immediately after reset"