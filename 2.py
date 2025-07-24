# Deep learning ANN using small dataset (no CSV)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# 1. Mini dataset (age, bmi, children, male, smoker)
# Format: [age, bmi, kids, male, smoker] → 1 = high cost, 0 = low cost
x = np.array([
    [19, 27.9, 0, 0, 1],
    [18, 33.7, 1, 1, 0],
    [28, 33.0, 3, 1, 0],
    [33, 22.7, 0, 1, 0],
    [32, 28.8, 0, 1, 0],
    [31, 25.7, 0, 0, 1],
    [46, 33.4, 1, 0, 1],
    [37, 27.7, 3, 1, 0],
    [60, 22.9, 0, 1, 0],
    [25, 28.0, 0, 1, 1]
])

# Labels: 1 = high charge, 0 = low charge
y = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1])

# 2. Scale features
sc = MinMaxScaler()
x = sc.fit_transform(x)

# 3. Split data
x1, x2, y1, y2 = train_test_split(x, y, test_size=0.3, random_state=1)

# 4. Build ANN model
m = Sequential()
m.add(Dense(8, activation='relu', input_shape=(5,)))
m.add(Dense(4, activation='relu'))
m.add(Dense(1, activation='sigmoid'))

# 5. Compile and train
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m.fit(x1, y1, epochs=100, verbose=0)

# 6. Predict and evaluate
p = m.predict(x2)
p = (p > 0.5).astype(int)
acc = accuracy_score(y2, p)

print("✅ Accuracy:", acc * 100, "%")
print("📊 Predicted:", p.flatten())
print("🎯 Actual:   ", y2)
