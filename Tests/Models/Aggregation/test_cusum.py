import pytest
import numpy as np

from Models.Aggregation import CusumTest, CusumAlert, CusumAlertDirection


@pytest.mark.parametrize('mu,slack,threshold', [
    (0.0, 0.1, 2.0),
    (0.0, 0.5, 2.0)
])

def test_cusum_test(mu, slack, threshold):
    sut = CusumTest(mu, slack, threshold)
    # test: positive growth
    data = np.linspace(0, threshold + 1, 10)
    exceedances = sut.run(data.tolist())
    results_growth = list(zip(data, exceedances))
    assert any((x for x in exceedances if x.is_critical)), "No event detected (positive branch)."

    sut = CusumTest(mu, slack, threshold)
    data = np.linspace(0, -threshold - 1, 10)
    exceedances = sut.run(data.tolist())
    result_shrink = list(zip(data, exceedances))
    assert any((x for x in exceedances if x.is_critical)), "No event detected (negative branch)."

    for d, e in results_growth:
        assert abs(e.c_minus) < 1e-8, "Negative term should not grow in positive branch."
        if e.c_plus > threshold:
            assert e.is_critical, "Criticality was not detected."
        else:
            assert not e.is_critical, "Detected critically when none was present."

    for d, e in result_shrink:
        assert abs(e.c_plus) < 1e-8, "Positive term should not grow in negative branch."
        if e.c_minus < -threshold:
            assert e.is_critical, "Criticality was not detected."
        else:
            assert not e.is_critical, "Detected critically when none was present."
