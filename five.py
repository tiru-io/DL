import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

np.random.seed(42)
s = (100, 100)
n = 1000
l = 0.1
c = np.random.random(s)
o = c + l * np.random.random(s)
x = o.flatten().reshape(-1, 1)
y = c.flatten()

m = MLPRegressor(max_iter=200, random_state=42)
p = {
    'hidden_layer_sizes': [(100,), (50, 50), (25, 25, 25)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}
g = GridSearchCV(m, p, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
g.fit(x, y)

b = g.best_params_
d = g.best_estimator_.predict(x).reshape(s)
e = mean_squared_error(c, d)
print("Best Hyperparameters:", b)
print("Mean Squared Error:", e)
