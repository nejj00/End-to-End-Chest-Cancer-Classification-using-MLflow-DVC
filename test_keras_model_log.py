import mlflow
import tensorflow as tf

model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
model.compile(optimizer='adam', loss='mse')
mlflow.keras.log_model(model, "simple_model")


# import tensorflow as tf

# model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
# model.compile(optimizer='adam', loss='mse')
# model.save("simple_model")