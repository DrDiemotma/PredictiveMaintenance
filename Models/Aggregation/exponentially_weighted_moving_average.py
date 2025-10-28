from dataclasses import dataclass
from enum import IntEnum
from .adaptive_filter_base import FilterBase

class EwmaDirection(IntEnum):
    """Direction of the EWMA threshold direction."""
    BOTH = 0
    """Process is is limited to both, upper and lower boundary."""
    UPPER_BOUNDARY = 1
    """Process is limited to upper but not lower boundary."""
    LOWER_BOUNDARY = -1
    """Process is limited to lower but not upper boundary."""


@dataclass(frozen=True)
class EwmaResult:
    """Result of an exponentially weighted moving average."""
    filtered_value: float
    """The filtered value."""
    exceeds_threshold: bool
    """Whether the value exceeds the preset threshold."""

class EwmaTest(FilterBase):
    """Exponentially weighted moving average process."""
    def __init__(self, initial_value: float, alpha: float, threshold: float,
                 direction: EwmaDirection = EwmaDirection.BOTH,
                 bias: float = 0.0):
        """
        ctor.
        :param initial_value: Initial value of the process.
        :param alpha: Adaption rate, (0..1].
        :param threshold: Threshold when an event is detected.
        :param direction: Direction to detect events.
        :param bias: Bias for the data to shift the process.
        """

        if alpha <= 0 or alpha > 1:
            raise ValueError(f"Valid interval for adaption value alpha: [0..1), was {alpha}.")
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, was {threshold}.")
        self._alpha: float = alpha
        self._threshold: float = threshold
        self._direction: EwmaDirection = direction
        self._bias: float = bias

        self._current_value: float = initial_value - bias

    def next(self, value: float):
        """
        Add the next value to the filter.
        :param value: Value to add to the filter.
        :return: The evaluation result.
        """
        self._current_value = self._alpha * (value - self._bias) + (1 - self._alpha) * self._current_value
        exceedance = False
        match self._direction:
            case EwmaDirection.BOTH:
                exceedance = abs(self._current_value) > self._threshold
            case EwmaDirection.UPPER_BOUNDARY:
                exceedance = self._current_value > self._threshold
            case EwmaDirection.LOWER_BOUNDARY:
                exceedance = self._current_value < -self._threshold

        return EwmaResult(filtered_value=self._current_value + self._bias, exceeds_threshold=exceedance)

    def reset(self, value: float = 0.0) -> None:
        """
        Reset the filter.
        :param value: Value to reset the filter to.
        :return: None.
        """
        self._current_value = value

