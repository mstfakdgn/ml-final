# mlp for binary classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

people = pd.read_csv('./breast-cancer-wisconsin.data')

df = people.copy()
df.fillna(df.mean(), inplace=True)

X = df.iloc[:, 1:10]
y = df.iloc[:, 10]

X = X.astype('float32')
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

n_features = X_train.shape[1]

model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Lost->',loss,'Accuracy->', acc)



