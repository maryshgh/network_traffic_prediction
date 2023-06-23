from tensorflow.keras import layers

# Construct the convLSTM Model
def convLSTM_model(inputs, filters, recurrent_length, r):
    x = layers.ConvLSTM2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
        data_format='channels_last',
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=filters[1], kernel_size=(3, 3, 3), padding="same")(x)
    x = layers.Reshape((recurrent_length, r, r))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=filters[3], kernel_size=(3, 3), padding="same", data_format="channels_first")(x)
    return x