import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Input

def f(n, t):
    d = np.sin(np.arange(0, n * t) * 0.1).reshape(n, t, 1)
    return d

n, t = 1000, 10
d = f(n, t)
s = int(0.8 * n)
r, e = d[:s], d[s:]
i, g = r[:, :-1, :], r[:, 1:, :]
x, y = e[:, :-1, :], e[:, 1:, :]

m = Sequential([
    Input(shape=(t - 1, 1)),
    SimpleRNN(32),
    Dense(1)
])
m.compile(loss='mean_squared_error', optimizer='adam')
m.fit(i, g, epochs=50, batch_size=32, verbose=0)
print("Test loss:", m.evaluate(x, y))

p = []
q = x[0:1, :, :]
for _ in range(10):
    z = m.predict(q, verbose=0)
    p.append(float(z[0, 0]))
    q = np.concatenate((q[:, 1:, :], z[:, np.newaxis, :]), axis=1)
print("Predictions:", p)
