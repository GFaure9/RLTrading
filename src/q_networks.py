import tensorflow as tf


def qnet1(input_shape: int, output_shape: int) -> tf.keras.Sequential:
    q_net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='linear'),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(output_shape, activation='relu')
    ])
    q_net.compile(optimizer="adam", loss="mse")
    return q_net





