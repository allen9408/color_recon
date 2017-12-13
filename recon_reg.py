import pandas as pd 
import numpy as np 
from keras.layers import Dense, Activation, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
df_input = pd.read_csv('input.csv')
input_data = df_input.as_matrix()

# df_label = pd.read_csv('color.csv')
df_label = pd.read_csv('color_labels.csv')
label = df_label.as_matrix()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11)

# import pdb
# pdb.set_trace()

model = Sequential()
model.add(Dense(27, input_dim=9, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(3, activation='relu'))
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=opt,  metrics=['mse'])

model.fit(X_train, y_train, epochs=200, batch_size = 100)

score = model.evaluate(X_test, y_test)
output = model.predict(input_data, batch_size = 100)
np.savetxt('out_reg.csv', output, delimiter=',')
print('Loss = ', score)