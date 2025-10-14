import pytest
import numpy as np
from Tests.Helper import data_generator, MarkovModelGenerator


@pytest.mark.parametrize("sequence_length,dimension,batch_size,sequence_count", [
    (10, 5, 15, 20),
    (11, 6, 16, 21)
])
def test_data_generator(sequence_length: int, dimension: int, batch_size: int, sequence_count: int):
    n_batches: int = 0
    for batch_tuple in data_generator(sequence_length, dimension, batch_size, sequence_count):
        assert len(batch_tuple[0]) == batch_size and len(batch_tuple[1]) == batch_size
        for sequence in batch_tuple[0]:
            assert len(sequence) == sequence_length
        for sequence in batch_tuple[1]:
            assert len(sequence) == sequence_length

        n_batches += 1

    assert n_batches == sequence_count


@pytest.mark.parametrize("sequence_length,dimension,batch_size,sequence_count", [
    (10, 5, 15, 20),
    (11, 6, 16, 21)
])
def test_data_generator_markov_integration(sequence_length: int, dimension: int, batch_size: int, sequence_count: int):
    transition_matrix: np.typing.NDArray = np.array([[0.8, 0.2, 0.0],
                                                     [0.1, 0.8, 0.1],
                                                     [0.3, 0.0, 0.7]],
                                                    dtype=np.float32)
    init_state: int = 0
    mkg: MarkovModelGenerator = MarkovModelGenerator(transition_matrix, init_state)
    data = data_generator(sequence_length, dimension, batch_size, sequence_count, model=mkg.get_next)
    assert data is not None