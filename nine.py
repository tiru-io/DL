import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

np.random.seed(0)
X = np.random.rand(1000, 100)
i = Input(shape=(X.shape[1],))
e = Dense(10, activation='relu')(i)
d = Dense(X.shape[1], activation='sigmoid')(e)
m = Model(i, d)
m.compile(optimizer='adam', loss='mse')
m.fit(X, X, epochs=50, batch_size=32)
n = Model(i, e)
f = n.predict(X)
print(f"Original shape: {X.shape}")
print(f"Encoded shape: {f.shape}")
