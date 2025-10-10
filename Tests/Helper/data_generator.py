import numpy as np
from typing import Generator

def data_generator(sequence_length: int, dimension: int, batch_size: int = 32, sequence_count: int = 50, seed: int = 42)\
        -> Generator[np.typing.NDArray[np.float32], None, None]:
    rng = np.random.default_rng(seed)

    for _ in range(sequence_count):
        current_batch = rng.standard_normal(size=(batch_size, sequence_length, dimension), dtype=np.float32)
        yield current_batch
