import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
from tensorflow.keras.datasets import cifar10
import numpy as np
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

def build_model(hp):
    model = models.Sequential([
        layers.Conv2D(filters=hp.Int('filters_1', 32, 128, step=32),
                      kernel_size=hp.Choice('kernel_1', [3, 5]),
                      activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=hp.Int('filters_2', 32, 128, step=32),
                      kernel_size=hp.Choice('kernel_2', [3, 5]),
                      activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(units=hp.Int('dense_units', 64, 256, step=64), activation='relu'),
        layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='cnn_tuning'
)

tuner.search(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

best_model = tuner.get_best_models(num_models=1)[0]
best_model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

uploaded = files.upload()

for filename in uploaded.keys():
    img = Image.open(filename)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Uploaded Image")
    plt.show()

img_resized = img.resize((32, 32))
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = best_model.predict(img_array)
predicted_class = np.argmax(predictions[0])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Predicted Class: {class_names[predicted_class]}")
