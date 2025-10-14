import tensorflow as tf
from tensorflow.keras import layers


def build_lstm_autoencoder(sequence_length: int,
                           feature_count: int,
                           latent_dim: int = 32,
                           dropout_rate: float = 0.2,
                           lstm1_output: int = 128,
                           lstm2_output: int = 64,
                           learning_rate: float = 1e-3) -> tf.keras.models.Model:
    """
    Configures an autoencoder based on LSTMs.
    :param sequence_length: The length of a sequence of vectors used for the analysis.
    :param feature_count: Number of features in each input vector.
    :param latent_dim: Size of the tight latent layer to encode the data.
    :param dropout_rate: Rate of the dropouts for training.
    :param lstm1_output: First layer output, and also for output size.
    :param lstm2_output: Second layer output. If 0, only one LSTM layer is used.
    :param learning_rate: How fast the algorithm adapts.
    :return: Compiled model which can be trained on data.
    :raises ValueError: if one parameter is not in of valid size.
    """
    if sequence_length < 1:
        raise ValueError("The sequence must have a valid length.")
    if feature_count < 1:
        raise ValueError("The dimension of the sequence must be greater than 1.")
    if latent_dim < 1:
        raise ValueError("The latent dimension must be positive.")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate must be between 0 and 1.")
    if lstm1_output < 1:
        raise ValueError("LSTM size for layer 1 must be positive.")

    input_layer = layers.Input(shape=(sequence_length, feature_count), name="input_layer")
    encoder_lstm1 = layers.LSTM(lstm1_output, return_sequences=True, name="encoder_lstm1")(input_layer)
    dropout_layer = layers.Dropout(dropout_rate, name="encoder_dropout")(encoder_lstm1)
    if lstm2_output > 0:
        encoder_lstm2 = layers.LSTM(lstm2_output, return_sequences=False)(dropout_layer)
        latent_layer = layers.Dense(latent_dim, activation="relu", name="latent_layer")(encoder_lstm2)
        decoder_dense = layers.Dense(sequence_length * lstm2_output, activation="relu", name="decoder_dense")(
            latent_layer)
        decoder_reshaped = layers.Reshape((sequence_length, lstm2_output), name="decoder_reshaped")(decoder_dense)
        decoder_lstm2 = layers.LSTM(lstm2_output, return_sequences=True, name="decoder_lstm2")(decoder_reshaped)
        decoder_lstm1 = layers.LSTM(lstm1_output, return_sequences=True, name="decoder_lstm1")(decoder_lstm2)
    else:
        latent_layer = layers.Dense(latent_dim, activation="relu", name="latent_layer")(dropout_layer)
        decoder_dense = layers.Dense(sequence_length * lstm1_output, activation="relu", name="decoder_dense")(
            latent_layer)
        decoder_reshaped = layers.Reshape((sequence_length, lstm1_output), name="decoder_reshaped")(decoder_dense)
        decoder_lstm1 = layers.LSTM(lstm1_output, return_sequences=True, name="decoder_lstm1")(decoder_reshaped)

    output_layer = layers.TimeDistributed(
        layers.Dense(feature_count, activation="linear"),
        name="reconstruction"
    )(decoder_lstm1)

    model = tf.keras.models.Model(input_layer, output_layer, name="LSTM_Autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    return model

