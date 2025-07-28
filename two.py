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

data = {
    'age': [19, 18, 28, 33, 32],
    'sex': ['female', 'male', 'male', 'male', 'male'],
    'bmi': [27.900, 33.770, 33.000, 22.705, 28.880],
    'children': [0, 1, 3, 0, 0],
    'smoker': ['yes', 'no', 'no', 'no', 'no'],
    'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest'],
    'charges': [16884.92400, 1725.55230, 4449.46200, 21984.47061, 3866.85520]
}
df = pd.DataFrame(data)
print("First few rows of the dataset:\n", df.head())
plt.figure(figsize=(8,5))
msno.matrix(df)
plt.show()
print("\nCounting missing values in each column:\n", df.isnull().sum())
print("\nSummary of the dataset (data types and non-null counts):\n", df.info())
print("\nSummary statistics of numerical columns:\n", df.describe())
print("\nFirst few rows of the dataset after initial analysis:\n", df.head())

df = pd.concat([df, pd.get_dummies(df['sex'], drop_first=True)], axis=1)
df = pd.concat([df, pd.get_dummies(df['smoker'], drop_first=True).rename(columns={'yes':'Smoker'})], axis=1)
print("\nUnique values in the 'region' column:\n", df['region'].unique())
df = pd.concat([df, pd.get_dummies(df['region'])], axis=1)
print("\nFirst few rows of the dataset after encoding categorical variables:\n", df.head())

plt.figure(figsize=(8,4))
sns.set_style('white')
sns.countplot(x='sex', data=df, palette='GnBu')
sns.despine(left=True)
plt.show()

plt.figure(figsize=(8,4))
sns.set_style('white')
sns.boxplot(x='sex', y='charges', data=df, palette='OrRd', hue='Smoker')
sns.despine(left=True)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
sns.scatterplot(x='age', y='charges', data=df, palette='coolwarm', hue='sex', ax=ax[0])
sns.scatterplot(x='age', y='charges', data=df, palette='GnBu', hue='Smoker', ax=ax[1])
sns.scatterplot(x='age', y='charges', data=df, palette='magma_r', hue='region', ax=ax[2])
sns.set_style('dark')
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
sns.boxplot(x='region', y='charges', data=df, palette='GnBu', hue='Smoker', ax=ax[0])
sns.boxplot(x='region', y='charges', data=df, palette='coolwarm', hue='sex', ax=ax[1])
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
sns.scatterplot(x='bmi', y='charges', data=df, palette='GnBu_r', hue='sex', ax=ax[0])
sns.scatterplot(x='bmi', y='charges', data=df, palette='magma', hue='Smoker', ax=ax[1])
sns.set_style('dark')
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

df.drop(['sex', 'region', 'smoker', 'southwest'], axis=1, inplace=True)
print("\nFirst few rows of the dataset after dropping unnecessary columns:\n", df.head())

plt.figure(figsize=(10,4))
sns.heatmap(df.corr(), cmap='OrRd')
plt.show()

X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_validate = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
model.fit(x=X_train, y=y_train, epochs=50, validation_data=(X_test, y_test), batch_size=128, callbacks=[early_stop])

loss = pd.DataFrame(model.history.history)
loss.plot()
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

pred = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred))
print(f"Root Mean Squared Error(RMSE) on Test Data: {rmse_test}")

entry_1 = df[:][1:3].drop('charges', axis=1)
pred = model.predict(entry_1)
rmse_entry_1 = np.sqrt(mean_squared_error(df[:][1:3]['charges'], pred))
