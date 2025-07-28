!pip install keras-tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from kerastuner.tuners import RandomSearch

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_conv_layers', 2, 4)):
        model.add(layers.Conv2D(hp.Int(f'conv_{i}_filters', 32, 256, 32), (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Int(f'dense_{i}_units', 64, 512, 32), activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, directory='hyperparameter_tuning', project_name='cifar10_tuner')
tuner.search(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
final_model = build_model(best_hps)
final_model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
loss, accuracy = final_model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')
