from dataclasses import dataclass
from enum import IntEnum


class EwmaDirection(IntEnum):
    BOTH = 0
    UPPER_BOUNDARY = 1
    LOWER_BOUNDARY = -1


@dataclass(frozen=True)
class EwmaResult:
    filtered_value: float
    exceeds_threshold: bool

class EwmaTest:
    def __init__(self, initial_value: float, alpha: float, threshold: float,
                 direction: EwmaDirection = EwmaDirection.BOTH,
                 bias: float = 0.0):

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

    def reset(self, value: float = 0.0):
        self._current_value = value

