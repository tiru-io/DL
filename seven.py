import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def loss(y_true, y_pred, t1=0.8, t2=1.2):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    tmp1 = (1 - y_true) * (y_pred ** (t1 - 1)) / t1
    tmp2 = (1 - y_true) * ((1 - y_pred) ** (t2 - 1)) / t2
    l = tf.reduce_mean(tf.reduce_sum(tmp1 + tmp2, axis=-1))
    return l

def net(shape):
    m = Sequential([Dense(64, activation='relu', input_shape=(shape,)), Dense(1, activation='sigmoid')])
    return m

np.random.seed(0)
X = np.random.rand(1000, 10)
y = (np.random.rand(1000) > 0.5).astype(np.float32)
m = net(10)
m.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
m.fit(X, y, epochs=10, batch_size=32)
l, a = m.evaluate(X, y)
print(f'Test loss: {l:.4f}, Test accuracy: {a*100:.2f}%')
