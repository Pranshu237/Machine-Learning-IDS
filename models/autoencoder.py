import tensorflow as tf
from tensorflow.keras import layers

def build_autoencoder(input_dim):
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(16, activation='relu')(encoded)

    decoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = tf.keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def train_autoencoder(X_train):
    model = build_autoencoder(X_train.shape[1])
    model.fit(X_train, X_train, epochs=10, batch_size=64, verbose=1)
    return model

def detect_anomaly(model, X, threshold=0.01):
    recon = model.predict(X)
    loss = ((X - recon) ** 2).mean(axis=1)
    return (loss > threshold).astype(int)