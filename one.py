import tensorflow as tf
from tensorflow.keras import layers as l, models as m, datasets as d
import matplotlib.pyplot as p

# GPU check
g = tf.config.list_physical_devices('GPU')
if g: print("✅ GPU:", g[0].name)

# Load & normalize
(x, y), (xv, yv) = d.cifar10.load_data()
x, xv = x / 255.0, xv / 255.0

# Model
n = m.Sequential([
    l.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    l.MaxPooling2D(2,2),
    l.Conv2D(64, (3,3), activation='relu', padding='same'),
    l.MaxPooling2D(2,2),
    l.Conv2D(128, (3,3), activation='relu', padding='same'),
    l.Flatten(),
    l.Dense(64, activation='relu'),
    l.Dense(10, activation='softmax')
])

# Compile
n.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
h = n.fit(x, y, epochs=10, batch_size=64, validation_data=(xv, yv))

# Accuracy Plot
p.plot(h.history['accuracy'], label='train')
p.plot(h.history['val_accuracy'], label='val')
p.title('Accuracy')
p.xlabel('Epoch')
p.ylabel('Acc')
p.legend()
p.grid(True)
p.show()

# Loss Plot
p.plot(h.history['loss'], label='train')
p.plot(h.history['val_loss'], label='val')
p.title('Loss')
p.xlabel('Epoch')
p.ylabel('Loss')
p.legend()
p.grid(True)
p.show()
