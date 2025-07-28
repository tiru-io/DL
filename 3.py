import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import cifar10
from kerastuner.tuners import RandomSearch

(x, y), (a, b) = cifar10.load_data()
x, a = x.astype('float32') / 255.0, a.astype('float32') / 255.0
y, b = tf.keras.utils.to_categorical(y, 10), tf.keras.utils.to_categorical(b, 10)

def f(h):
    m = Sequential()
    for i in range(h.Int('n', 2, 4)):
        m.add(layers.Conv2D(h.Int(f'c_{i}_f', 32, 256, 32), (3, 3), activation='relu', padding='same'))
        m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Flatten())
    for i in range(h.Int('d', 1, 3)):
        m.add(layers.Dense(h.Int(f'e_{i}_u', 64, 512, 32), activation='relu'))
    m.add(layers.Dense(10, activation='softmax'))
    m.compile(optimizer=tf.keras.optimizers.Adam(h.Choice('r', [1e-3, 1e-4])), 
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m

t = RandomSearch(f, objective='val_accuracy', max_trials=5, directory='h', project_name='c')
t.search(x, y, epochs=1, validation_data=(a, b))
p = t.get_best_hyperparameters(num_trials=1)[0]
n = f(p)
n.fit(x, y, epochs=1, validation_data=(a, b))
l, c = n.evaluate(a, b)
print(f'Test accuracy: {c}')
