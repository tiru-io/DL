import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

d = {
    'age': [19, 18, 28, 33, 32],
    'sex': ['female', 'male', 'male', 'male', 'male'],
    'bmi': [27.9, 33.77, 33.0, 22.705, 28.88],
    'child': [0, 1, 3, 0, 0],
    'smk': ['yes', 'no', 'no', 'no', 'no'],
    'reg': ['sw', 'se', 'se', 'nw', 'nw'],
    'chg': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552]
}
df = pd.DataFrame(d)

plt.figure(figsize=(8,5))
msno.matrix(df)
plt.show()
print("\nMissing values:\n", df.isnull().sum())
print("\nInfo:\n")
df.info()
print("\nDescribe:\n", df.describe())

df = pd.concat([df, pd.get_dummies(df['sex'], drop_first=True)], axis=1)
df = pd.concat([df, pd.get_dummies(df['smk'], drop_first=True).rename(columns={'yes': 'smk_y'})], axis=1)
print("\nRegions:\n", df['reg'].unique())
df = pd.concat([df, pd.get_dummies(df['reg'])], axis=1)

plt.figure(figsize=(8,4))
sns.countplot(x='sex', data=df, palette='GnBu')
sns.despine(left=True)
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x='sex', y='chg', data=df, palette='OrRd', hue='smk_y')
sns.despine(left=True)
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12,5))
sns.scatterplot(x='age', y='chg', data=df, hue='sex', ax=ax[0])
sns.scatterplot(x='age', y='chg', data=df, hue='smk_y', ax=ax[1])
sns.scatterplot(x='age', y='chg', data=df, hue='reg', ax=ax[2])
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.boxplot(x='reg', y='chg', data=df, hue='smk_y', ax=ax[0])
sns.boxplot(x='reg', y='chg', data=df, hue='sex', ax=ax[1])
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.scatterplot(x='bmi', y='chg', data=df, hue='sex', ax=ax[0])
sns.scatterplot(x='bmi', y='chg', data=df, hue='smk_y', ax=ax[1])
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

df.drop(['sex', 'reg', 'smk', 'sw'], axis=1, inplace=True)
print("\nCleaned df:\n", df.head())

plt.figure(figsize=(10,4))
sns.heatmap(df.corr(), cmap='OrRd')
plt.show()

X = df.drop('chg', axis=1)
y = df['chg']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25)

sc = MinMaxScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

m = Sequential()
m.add(Dense(8, activation='relu'))
m.add(Dense(3, activation='relu'))
m.add(Dense(1))
m.compile(optimizer='adam', loss='mse')

cb = EarlyStopping(monitor='val_loss', mode='min', patience=15)
m.fit(x=X_tr, y=y_tr, epochs=50, validation_data=(X_te, y_te), batch_size=128, callbacks=[cb])

pd.DataFrame(m.history.history).plot()
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

pred = m.predict(X_te)
rmse = np.sqrt(mean_squared_error(y_te, pred))
print(f"\nTest RMSE: {rmse:.2f}")

x_new = df[1:3].drop('chg', axis=1)
y_true = df[1:3]['chg']
p_new = m.predict(x_new)
rmse_new = np.sqrt(mean_squared_error(y_true, p_new))
print(f"\nNew Data RMSE: {rmse_new:.2f}")
