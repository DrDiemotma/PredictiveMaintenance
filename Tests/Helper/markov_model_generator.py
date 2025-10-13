import numpy as np
from numpy.random import Generator

SEED: int = 42  # seed when the init vector is not provided

class MarkovModelGenerator:
    def __init__(self, transition_matrix: np.typing.NDArray, init_state: int):
        self._state = init_state
        self._transition_matrix = transition_matrix
        self._dimension = len(transition_matrix)
        self._mode_centers: np.typing.NDArray = np.linspace(0, self._dimension, self._dimension)
        self._states = [self._state]

    def get_next(self, batch_size: int, sequence_length: int, dimension, rng: Generator) -> np.typing.NDArray:
        assert dimension == self._dimension
        batches = []
        for _ in range(batch_size):
            batch = []
            for _ in range(sequence_length):
                measurement: np.typing.NDArray = rng.standard_normal(self._dimension) + self._mode_centers[self._state]
                self._next_state(rng)
                batch.append(measurement)
            batches.append(batch)
        return np.stack(batches, dtype=np.float32)

    def _next_state(self, rng: Generator):
        transition_probabilities = self._transition_matrix[self._state]
        self._state = rng.choice(self._dimension, p=transition_probabilities/np.sum(transition_probabilities))
        self._states.append(self._state)
