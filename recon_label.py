import numpy as np 
import pandas as pd 
import pdb

df_input = pd.read_csv('input.csv')
X = df_input.as_matrix()

df_label = pd.read_csv('color_labels_6.csv')
# df_label = pd.read_csv('color_labels.csv')
output = df_label.as_matrix()
centroids = np.unique(output, axis=0)
num_label = centroids.shape[0]
y = [np.where(output[i]==centroids)[0].tolist()[0] for i in range(output.shape[0])]
y = np.array(y)
y_onehot = np.zeros((y.shape[0], num_label))
y_onehot[np.arange(y.shape[0]), y] = 1

# pdb.set_trace()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_onehot,test_size=0.2)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(27, input_dim = 9, activation = 'relu'))
model.add(Dense(18, activation = 'relu'))
model.add(Dense(9, activation = 'relu'))
model.add(Dense(num_label, activation = 'softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 100, epochs = 150)

score = model.evaluate(X_test, y_test)

y_predict = model.predict(X).tolist()
label_predict = [i.index(max(i)) for i in y_predict]
# pdb.set_trace()
y_out = np.array([centroids.tolist()[i] for i in label_predict])
np.savetxt('output.csv',y_out, delimiter=',')

print('accuracy = ', score)