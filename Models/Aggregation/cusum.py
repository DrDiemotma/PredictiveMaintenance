from enum import IntEnum
from dataclasses import dataclass
from .adaptive_filter_base import FilterBase

class CusumAlertDirection(IntEnum):
    """
    Provides the alert direction
    """
    NONE = 0
    """No direction (no alert)."""
    PLUS = 1
    """Positive direction (sequence exceeds threshold)."""
    MINUS = -1


@dataclass(frozen=True)
class CusumAlert:
    """
    Return type for a CUSUM alert.
    """
    direction: CusumAlertDirection
    """Direction for al alert."""
    c_plus: float
    """Current C+ value."""
    c_minus: float
    """Current C- value."""
    is_critical: bool
    """Whether this even was critical."""


class CusumTest(FilterBase):
    """
    Cumulative sum test for an event based setup.
    """
    def __init__(self, mu: float, slack: float, threshold: float):
        """
        Ctor.
        :param mu: Expected mean of the signal.
        :param slack: Slack variable for the process.
        :param threshold: Threshold when the system is supposed to rise an alert.
        """
        self._c_plus: float = 0.0
        self._c_minus: float = 0.0
        self._threshold = threshold
        self._mu = mu
        self._slack = slack

    @property
    def mu(self):
        """Mean value of the filter."""
        return self._mu

    @mu.setter
    def mu(self, value):
        """Mean value of the filter."""
        self._mu = value

    @property
    def slack(self):
        """Slack of the filter."""
        return self._slack

    @slack.setter
    def slack(self, value):
        """Slack of the filter."""
        self._slack = value

    def next(self, value: float) -> CusumAlert:
        """
        Add the next value to the filter.
        :param value: The value to add.
        :return: Description of the evaluation.
        """
        self._c_plus = max(0.0, self._c_plus + (value - self._mu - self._slack))
        self._c_minus = min(0.0, self._c_minus + (value - self._mu + self._slack))

        if self._c_plus > self._threshold:
            alert = CusumAlert(
                direction=CusumAlertDirection.PLUS,
                c_plus=self._c_plus,
                c_minus=self._c_minus,
                is_critical=True
            )
            self._c_plus = 0.0
            return alert

        if self._c_minus < -self._threshold:
            alert = CusumAlert(
                direction=CusumAlertDirection.MINUS,
                c_plus=self._c_plus,
                c_minus=self._c_minus,
                is_critical=True
            )
            self._c_minus = 0.0
            return alert

        return CusumAlert(
            direction=CusumAlertDirection.NONE,
            c_plus=self._c_plus,
            c_minus=self._c_minus,
            is_critical=False
        )


