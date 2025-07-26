import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Step 1: Simulate Data ---

# Constants
num_samples = 1500
image_shape = (32, 32, 3)
num_classes = 3

# Generate random image data (values between 0 and 1)
X = np.random.rand(num_samples, *image_shape)

# Generate random labels (0 = Young, 1 = Middle, 2 = Old)
y = np.random.randint(0, num_classes, num_samples)
y = to_categorical(y, num_classes=num_classes)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 2: Build ANN Model ---

model = Sequential()

# Flatten 32x32x3 = 3072 inputs
model.add(Flatten(input_shape=image_shape))

# Hidden layer with 500 nodes and ReLU activation
model.add(Dense(500, activation='relu'))

# Output layer with 3 nodes and Softmax activation
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Step 3: Train the Model ---

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,
                    batch_size=64)

# --- Step 4: Plot Accuracy and Loss ---

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='x')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# --- Step 5: Predict a Random Sample ---

index = np.random.randint(0, len(X_val))
sample = X_val[index]
true_label = np.argmax(y_val[index])
predicted_label = np.argmax(model.predict(np.expand_dims(sample, axis=0)))

labels = ["Young", "Middle", "Old"]
print("Actual:", labels[true_label])
print("Predicted:", labels[predicted_label])

# Display the sample image
plt.imshow(sample)
plt.title(f"Actual: {labels[true_label]} / Predicted: {labels[predicted_label]}")
plt.axis('off')
plt.show()
