import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
def alexnet(s, c):
    m = Sequential([
        Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=s, padding='valid'),
        MaxPooling2D((3,3), strides=(2,2), padding='valid'),
        Conv2D(256, (5,5), activation='relu', padding='same'),
        MaxPooling2D((3,3), strides=(2,2), padding='valid'),
        Conv2D(384, (3,3), activation='relu', padding='same'),
        Conv2D(384, (3,3), activation='relu', padding='same'),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        MaxPooling2D((3,3), strides=(2,2), padding='valid'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(c, activation='softmax')
    ])
    return m
s = (227, 227, 3)
c = 1000
m = alexnet(s, c)
m.summary()
