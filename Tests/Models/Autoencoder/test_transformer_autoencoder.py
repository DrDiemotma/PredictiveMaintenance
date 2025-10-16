import pytest
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras.models import Model
from Models.Autoencoder import build_transformer_autoencoder, make_tf_dataset
from Tests.Helper import MarkovModelGenerator, data_generator


@pytest.mark.parametrize("sequence_length,dimension,batch_size,sequence_count", [
    (10, 5, 15, 20),
    (11, 6, 16, 21)
])
def test_transformer_autoencoder(sequence_length, dimension, batch_size, sequence_count):
    transition_matrix: np.typing.NDArray = np.array([[0.8, 0.2, 0.0],
                                                     [0.1, 0.8, 0.1],
                                                     [0.3, 0.0, 0.7]],
                                                    dtype=np.float32)
    init_state = 0
    mkg: MarkovModelGenerator = MarkovModelGenerator(transition_matrix, init_state)
    data_gen = lambda: data_generator(sequence_length, dimension, batch_size, sequence_count, model=mkg.get_next)
    dataset: Dataset = make_tf_dataset(data_gen, sequence_length, dimension, batch_size)
    transformer_autoencoder: Model = build_transformer_autoencoder(sequence_length, dimension)
    history = transformer_autoencoder.fit(dataset, epochs=1, steps_per_epoch=1)
    final_loss = history.history["loss"][-1]
    assert final_loss < float('inf'), "Deviation was not able to be measured"
