import tensorflow as tf
import numpy as np
from typing import Callable, Generator

def make_tf_dataset(generator_function: Callable[[], Generator[tuple[np.typing.NDArray, np.typing.NDArray], None, None]],
                    sequence_length: int,
                    feature_count: int,
                    batch_size: int) -> tf.data.Dataset:
    """
    Wraps the generator to a TensorFlow dataset to train autoencoder.
    :param generator_function: Generator function to wrap.
    :param sequence_length: Length of a sequence.
    :param feature_count: Dimension of each vector.
    :param batch_size: Batch size (sequences per batch).
    :return: Dataset to train an autoencoder on.
    """
    output_signature: tuple[tf.TensorSpec, tf.TensorSpec] = (
        tf.TensorSpec(shape=(batch_size, sequence_length, feature_count), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, sequence_length, feature_count), dtype=tf.float32)
    )

    return tf.data.Dataset.from_generator(generator_function, output_signature=output_signature)