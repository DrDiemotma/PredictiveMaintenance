import numpy as np
from typing import Generator, Callable
from numpy.random import Generator as RandomGenerator

GeneratorModel = Callable[[int, int, int, RandomGenerator], np.typing.NDArray]

def data_generator(sequence_length: int, dimension: int, batch_size: int = 32, sequence_count: int = 50,
                   seed: int = 42, model: GeneratorModel = None)\
        -> Generator[tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]], None, None]:
    rng: RandomGenerator = np.random.default_rng(seed)
    if model is None:
        model = lambda bs, sl, d, rn: rn.standard_normal(size=(batch_size, sequence_length, dimension), dtype=np.float32)

    for _ in range(sequence_count):
        current_batch = model(batch_size, sequence_length, dimension, rng)
        yield current_batch, current_batch
