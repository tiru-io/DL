import tensorflow as tf
import tensorflow_datasets as tfds

# Load and prepare dataset
ds, info = tfds.load('cifar10', as_supervised=True, with_info=True)
tr, ts = ds['train'], ds['test']

def prep(x, y):
    x = tf.image.resize(x, (32, 32))
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

bs = 32
tr = tr.map(prep).shuffle(50000).batch(bs).prefetch(tf.data.AUTOTUNE)
ts = ts.map(prep).batch(bs).prefetch(tf.data.AUTOTUNE)

# Build model
def cnn(inp, cls):
    m = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=inp),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(cls, activation='softmax')
    ])
    return m

inp = (32, 32, 3)
cls = 10
m = cnn(inp, cls)
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train & evaluate
m.fit(tr, epochs=10)
loss, acc = m.evaluate(ts)
print(f"Test accuracy: {acc}")
