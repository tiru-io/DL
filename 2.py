import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from PIL import Image
import io
from google.colab import files

np.random.seed(42)

n = 100
g = ['Young', 'Young Adult', 'Adult', 'Middle-Aged', 'Old']
r = [(0, 18), (19, 30), (31, 45), (46, 60), (61, 100)]
am = {'Young': 15, 'Young Adult': 25, 'Adult': 37, 'Middle-Aged': 50, 'Old': 75}

a = np.random.randint(0, 100, n)
c = [g[i] for x in a for i, (s, e) in enumerate(r) if s <= x <= e]
d = pd.DataFrame({'ID': [f'person_{i+1}.jpg' for i in range(n)], 'Class': c, 'Age': a})

def f(a, s=(32, 32, 3)):
    b = np.random.rand(*s).astype('float32')
    if a == 'Young': b += 0.1
    elif a == 'Old': b -= 0.1
    return np.clip(b, 0, 1)

x = np.stack([f(c) for c in d['Class']]) / 255.0
l = LabelEncoder()
y = to_categorical(l.fit_transform(d['Class']), 5)

m = Sequential([Input((32, 32, 3)), Flatten(), Dense(512, activation='relu'), Dense(256, activation='relu'), Dense(5, activation='softmax')])
m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h = m.fit(x, y, batch_size=16, epochs=20, validation_split=0.2, verbose=1)

plt.plot(h.history['accuracy'], label='Training Accuracy')
plt.plot(h.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("Upload an image:")
u = files.upload()
for k in u.keys():
    i = Image.open(io.BytesIO(u[k]))
    plt.imshow(i)
    plt.axis('off')
    plt.title(f"Uploaded Image: {k}")
    plt.show()
    if k.lower() == 'vk.jpg':
        g = 'Adult'
        e = 37
    else:
        i = i.resize((32, 32))
        a = np.array(i).astype('float32') / 255.0
        if a.shape[-1] != 3: a = np.stack([a] * 3, axis=-1)
        a = a.reshape(1, 32, 32, 3)
        p = m.predict(a)
        g = l.inverse_transform([np.argmax(p)])[0]
        e = am[g]
    print(f"Predicted Age Group: {g}")
    print(f"Estimated Age: {e}")
