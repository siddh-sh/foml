# save_model.py
import tensorflow as tf
 
# Example: A simple trained model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
 
# Dummy training
import numpy as np
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, size=(100, 1))
model.fit(X, y, epochs=5)
 
# Save model
model.save("my_model.h5")
 