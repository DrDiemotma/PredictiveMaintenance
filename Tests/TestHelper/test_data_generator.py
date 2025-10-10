import pytest
from Tests.Helper import data_generator

@pytest.mark.parametrize("sequence_length,dimension,batch_size,sequence_count", [
    (10, 5, 15, 20),
    (11, 6, 16, 21)
])
def test_data_generator(sequence_length: int, dimension: int, batch_size: int, sequence_count: int):
    n_batches: int = 0
    for batch in data_generator(sequence_length, dimension, batch_size, sequence_count):
        assert len(batch) == batch_size
        for sequence in batch:
            assert len(sequence) == sequence_length

        n_batches += 1

    assert n_batches == sequence_count
