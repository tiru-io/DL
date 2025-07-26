# Install required libraries if needed
# !pip install tensorflow keras matplotlib

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset directly from Keras
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, y_train_cat, validation_split=0.2, epochs=5, batch_size=64)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f'\nTest Accuracy: {test_acc:.4f}')

# Predict on few test images
preds = model.predict(X_test[:10])
pred_labels = np.argmax(preds, axis=1)

# Show predictions
plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(f"P: {class_names[pred_labels[i]]}\nA: {class_names[int(y_test[i])]}")
plt.suptitle("Predicted vs Actual")
plt.tight_layout()
plt.show()
