import tensorflow as tf
from tensorflow.keras import layers


def positional_encoding(sequence_length: int, embedding_size: int) -> tf.Tensor:
    """
    Precure a sinosidal positional encoding.
    :param sequence_length: Length of a sequence to analyze.
    :param embedding_size: Size of embeddings.
    :return: The positional encoder
    """

    position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
    index = tf.range(embedding_size, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (index // 2)) / tf.cast(embedding_size, tf.float32))
    angle_rads = position * angle_rates  # angle_rads[i, j] = position[i] * angle_rates[j]
    sines = tf.sin(angle_rads[:, 0::2])
    coses = tf.cos(angle_rads[:, 1::2])
    encoding = tf.reshape(tf.concat([sines, coses], axis=-1), (sequence_length, embedding_size))
    return tf.cast(encoding, dtype=tf.float32)


def transformer_encoder_block(x: tf.Tensor,
                              embedding_size: int,
                              attention_head_counts: int,
                              dff: int,
                              dropout_rate: float,
                              name: str,
                              epsilon=1e-6) -> tf.Tensor:
    attn1 = layers.MultiHeadAttention(num_heads=attention_head_counts, key_dim=embedding_size // attention_head_counts,
                                      name=f"{name}_self_multi_headed_attention")(x, x, x)
    dropout1 = layers.Dropout(dropout_rate, name=f"{name}_self_dropout")(attn1)
    out1 = layers.LayerNormalization(epsilon=epsilon, name=f"{name}_self_layer_normalization")(x + dropout1)

    ffn1 = layers.Dense(dff, activation="relu", name=f"{name}_feed_forward_net1")(out1)
    ffn2 = layers.Dense(embedding_size, name=f"{name}_feed_forward_net2")(ffn1)
    ffn_dropout = layers.Dropout(dropout_rate, name=f"{name}_feed_forward_dropout")(ffn2)
    out2 = layers.LayerNormalization(epsilon=epsilon, name=f"{name}_feed_forward_layer_normalization")(out1 + ffn_dropout)
    return out2


def transformer_decoder_block(x: tf.Tensor,
                              enc_output: tf.Tensor,
                              embedding_size: int,
                              attention_head_counts: int,
                              dff: int,
                              dropout_rate: float,
                              name: str,
                              epsilon: float = 1e-6):
    attn1 = layers.MultiHeadAttention(num_heads=attention_head_counts, key_dim=embedding_size // attention_head_counts,
                                      name=f"{name}_self_multi_head_attention")(x, x, x)
    dropout1 = layers.Dropout(dropout_rate, name=f"{name}_self_dropout")(attn1)
    out1 = layers.LayerNormalization(epsilon=epsilon, name=f"{name}_self_layer_normalization")(x + dropout1)

    attn2 = layers.MultiHeadAttention(num_heads=attention_head_counts, key_dim=embedding_size // attention_head_counts,
                                      name=f"{name}_cross_multi_head_attention")(out1, enc_output, enc_output)
    dropout2 = layers.Dropout(dropout_rate, name=f"{name}_cross_dropout")(attn2)
    out2 = layers.LayerNormalization(epsilon=epsilon, name=f"{name}_cross_layer_normalization")(out1 + dropout2)

    ffn1 = layers.Dense(dff, activation="relu", name=f"{name}_ffn_1")(out2)
    ffn2 = layers.Dense(embedding_size, name=f"{name}_ffn_2")(ffn1)
    out3 = layers.LayerNormalization(epsilon=epsilon, name=f"{name}_ffn_layer_normalization")(out2 + ffn2)
    return out3


def build_transformer_autoencoder(sequence_length: int,
                                  feature_count: int,
                                  embedding_size: int = 64,
                                  attention_head_counts: int = 4,
                                  dff: int = 128,
                                  num_encoder_layers: int = 2,
                                  num_decoder_layers: int = 2,
                                  latent_dim: int = 32,
                                  dropout_rate: float = 0.2,
                                  learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build a model for a transformer autoencoder. This implementation utilizes a "bottleneck" to be more sensitive to
    unprecedented observations. This makes it more suitable as an event detector than a generative model.
    :param sequence_length: Length of an analyzed sequence.
    :param feature_count: Dimension of the measurements.
    :param embedding_size: Embedding dimension for the attention heads.
    :param attention_head_counts: Head count for the multi layer heads.
    :param dff: dimension of the feed forward networks.
    :param num_encoder_layers: Number of encoder layers.
    :param num_decoder_layers: Number of decoder layers.
    :param latent_dim: Size of the latent dimension (bottleneck).
    :param dropout_rate: Dropout rate used in different
    :param learning_rate: Adaption rate in each step.
    :return: A model to be trained on data.
    :raises ValueError: When one parameter is out of valid spaces.
    """
    if (embedding_size % attention_head_counts != 0) or (embedding_size < attention_head_counts):
        raise ValueError("The embedding size must be divisible by the number of attention heads.")
    if sequence_length < 1:
        raise ValueError("The sequence must have a positive length.")
    if feature_count < 1:
        raise ValueError("The dimension of the measurements must be at least one dimensional.")
    if embedding_size < 1:
        raise ValueError("The embedding size must be positive.")
    if attention_head_counts < 1:
        raise ValueError("The number of attention heads must be positive.")
    if dff < 1:
        raise ValueError("The dimension of the feed forward network must be positive.")
    if num_encoder_layers < 1:
        raise ValueError("At least one encoder layer must be used.")
    if num_decoder_layers < 1:
        raise ValueError("At least one decoder layer must be used.")
    if latent_dim < 1:
        raise ValueError("The size of the bottleneck must be positive.")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("The dropout rate must be between 0 and 1.")
    if learning_rate <= 0:
        raise ValueError("The learning rate must be positive.")

    inputs = layers.Input(shape=(sequence_length, feature_count), name="input_layer")
    encoding = tf.constant(positional_encoding(sequence_length, embedding_size))
    input_projection = layers.Dense(embedding_size, name="input_projection")(inputs) + encoding[tf.newaxis, :, :]
    input_dropout = layers.Dropout(dropout_rate, name="input_dropout")(input_projection)

    encoder_output = input_dropout
    for i in range(0, num_encoder_layers):
        encoder_output = transformer_encoder_block(
            encoder_output, embedding_size, attention_head_counts, dff, dropout_rate, f"encoder_{i}"
        )

    # make a bottleneck here -> layers.GlobalAveragePooling1D -> Dense(latent_dim, activation="relu")
    gap = layers.GlobalAvgPool1D(name="latent_pooling")(encoder_output)
    encoder_output = layers.Dense(latent_dim, activation="relu", name="latent_layer")(gap)
    encoder_output = layers.Dense(embedding_size, activation="relu", name="latent_layer_extension")(encoder_output)
    expanded = layers.Lambda(lambda t: tf.repeat(t[:, tf.newaxis, :], repeats=sequence_length, axis=1),
                             name="latent_broadcast")(encoder_output)



    decoder_input_projection = layers.Dense(embedding_size, name="decoder_input_projection")(expanded) \
        + encoding[tf.newaxis, :, :]
    decoder_input_dropout = layers.Dropout(dropout_rate, name="decoder_dropout")(decoder_input_projection)
    decoder_output = decoder_input_dropout
    encoder_expanded_for_mha = layers.Lambda(lambda t: tf.expand_dims(t, axis=1), name="encoder_expanded_for_mha")(
        encoder_output)
    for i in range(0, num_decoder_layers):
        decoder_output = transformer_decoder_block(decoder_output, encoder_expanded_for_mha, embedding_size,
                                                   attention_head_counts, dff, dropout_rate, f"decoder_{i}")

    outputs = layers.TimeDistributed(layers.Dense(feature_count, activation="linear"), name="reconstruction")(decoder_output)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="Transformer_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    return model
