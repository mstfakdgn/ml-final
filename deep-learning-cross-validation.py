# Use scikit-learn to grid search the learning rate and momentum
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

def create_model(learn_rate=0.01, momentum=0):
    
	model = Sequential()
	model.add(Dense(12, input_dim=9, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	
	optimizer = SGD(lr=learn_rate, momentum=momentum)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

seed = 7
np.random.seed(seed)

people = pd.read_csv('./breast-cancer-wisconsin.data')

df = people.copy()
df.fillna(df.mean(), inplace=True)

X = df.iloc[:, 1:10]
y = df.iloc[:, 10]

X = X.astype('float32')
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# define the grid search parameters
params = {
    "learn_rate" : [0.001, 0.01, 0.1, 0.2, 0.3],
    "momentum" : [0.0, 0.2, 0.4, 0.6, 0.9, 0.9]
}

model_cv = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=10, verbose=2).fit(X_train, y_train)

print("Best: %f using %s" % (model_cv.best_score_, model_cv.best_params_))


