import pytest
import numpy as np
from Models.Predictors import AutoencoderPredictor, AutoencoderType
from Tests.Helper import MarkovModelGenerator, data_generator


@pytest.mark.parametrize("sequence_length,dimension,batch_size,sequence_count", [
    (10, 5, 15, 20),
    (11, 6, 16, 21)
])
def test_autoencoder_predictor(sequence_length, dimension, batch_size, sequence_count):
    sut = AutoencoderPredictor(sequence_length, dimension, batch_size, AutoencoderType.GRU)
    transition_matrix: np.typing.NDArray = np.array([[0.8, 0.2, 0.0],
                                                     [0.1, 0.8, 0.1],
                                                     [0.3, 0.0, 0.7]],
                                                    dtype=np.float32)
    init_state: int = 0
    mkg: MarkovModelGenerator = MarkovModelGenerator(transition_matrix, init_state)
    data_gen = lambda: data_generator(sequence_length, dimension, batch_size, sequence_count, model=mkg.get_next)
    history = sut.fit(data_gen, epochs=1, steps_per_epoch=1)
    final_loss = history.history["loss"][-1]
    assert final_loss < float('inf'), "Deviation was not able to be measured."

    false_positives = 0
    counters = 0
    for test_data, _ in data_gen():
        predictions, _ = sut.predict(test_data)
        false_positives += sum(predictions)
        counters += len(predictions)
    prediction_limit = 0.05 * counters
    assert false_positives <= prediction_limit, f"Too many false positives in this test scenario: expected {prediction_limit} but was {false_positives}."
