from collections.abc import Callable
from tensorflow.keras.models import Model
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import History
import numpy as np
from .autoencoder_enum import AutoencoderType
from .. import Autoencoder

class AutoencoderPredictor:
    def __init__(self, sequence_length: int, feature_count: int, batch_size: int,
                 encoder: AutoencoderType = AutoencoderType.GRU, *args, **kwargs):
        self._sequence_length: int = sequence_length
        self._feature_count: int = feature_count
        self._batch_size = batch_size
        self._history: History | None = None
        self._model: Model
        self._mu: float = 0.0
        self._sigma: float = 0.0
        match encoder:
            case AutoencoderType.GRU:
                self._model = Autoencoder.build_gru_autoencoder(sequence_length, feature_count, *args, **kwargs)
            case AutoencoderType.LSTM:
                self._model = Autoencoder.build_lstm_autoencoder(sequence_length, feature_count, *args, **kwargs)
            case AutoencoderType.TRANSFORMER:
                self._model = Autoencoder.build_transformer_autoencoder(sequence_length, feature_count, *args, **kwargs)
            case _:
                raise NotImplementedError(f"The model for {encoder} is not implemented yet.")

    @property
    def trained(self):
        return self._history is not None

    def fit(self, data_generator: Callable, *args, **kwargs) -> History:
        """
        Fit the model to the provided data.
        :param data_generator: Generator for the data.
        :param args: args for the model fit. Uses TensorFlow notation.
        :param kwargs: kwargs for the model fit. Uses TensorFlow notation.
        :return: History of the training.
        """
        dataset: Dataset = Autoencoder.make_tf_dataset(data_generator, self._sequence_length, self._feature_count, self._batch_size)
        self._history = self._model.fit(dataset, *args, **kwargs)

        fresh_gen = data_generator()

        all_errors = []
        for data in fresh_gen:
            all_errors.extend(self._deviation(data[0]))

        self._mu = np.mean(all_errors)
        self._sigma = np.std(all_errors)
        return self._history


    def predict(self, data, threshold = 1.96) -> None | tuple[list[bool], list[float]]:
        """
        Predict the deviations from the expected sequence.
        :param data: Data for the deviation calculation.
        :param threshold:
        :return: List o
        """
        if not self.trained:
            return None

        deviation_sequence = self._deviation(data)
        z_scores = (deviation_sequence - self._mu) / self._sigma
        exceedance = z_scores > threshold
        return exceedance, z_scores

    def _predict(self, data):
        predicted_sequence = self._model.predict(data)
        return predicted_sequence

    def _deviation(self, data):
        predictions = self._predict(data)
        mse = np.mean(np.square(data - predictions), axis=(1, 2))
        return mse