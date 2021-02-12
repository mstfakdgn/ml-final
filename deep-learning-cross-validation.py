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
from sklearn.metrics import accuracy_score

def create_model(learn_rate=0.1, epochs = 150, batch_size=32, optimizer="rmsprop"):
    
	model = Sequential()
	model.add(Dense(12, input_dim=9, activation='relu'))
	# model.add(Dense(512,kernel_initializer=init))
	# model.add(Dense(10,kernel_initializer=init))
	model.add(Dense(1, activation='sigmoid'))
	
	# optimizer = SGD(lr=learn_rate, momentum=momentum)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = create_model()
model.fit(X_train, y_train)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy))
print('Loss: %f' % (loss))

model = KerasClassifier(build_fn=create_model)

params = {
    "learn_rate" : [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8, 1E-9],
	"epochs" : [50, 100, 150],
	"batch_size" : [5, 10, 20],
	"optimizer" : ['rmsprop', 'adam'],
	# "init" : ['glorot_uniform', 'normal', 'uniform']
}

model_cv = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=5, verbose=2).fit(X_train, y_train)

print("Best: %f using %s" % (model_cv.best_score_, model_cv.best_params_))

tuned_model = create_model(batch_size=10, epochs=50, learn_rate=1e-07, optimizer='rmsprop')
tuned_model.fit(X_train, y_train)

loss, accuracy = tuned_model.evaluate(X_test, y_test, verbose=0)
print('Tuned  Accuracy: %f' % (accuracy))
print('Tuned Loss: %f' % (loss))