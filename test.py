import tensorflow as tf

print("TensorFlow version:", tf.__version__)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("Keras import successful")

# Create a simple Sequential model
model = Sequential()
model.add(LSTM(50, input_shape=(10, 64)))  # Example shape (timesteps, features)
model.add(Dense(1))  # Output layer
print("Model created successfully")

