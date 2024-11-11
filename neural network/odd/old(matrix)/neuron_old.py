import numpy as np
from sklearn.model_selection import train_test_split

from math import factorial

from dataset_generator import max_size

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.optimizers import Adam


model = Sequential()

input_data = np.load('odd/input_data.npy')
# input_data = input_data.reshape((max_size*number_of_samples, max_size*max_size))
output_data = np.load('odd/output_data.npy')

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data,
                                                    train_size=0.67,
                                                    random_state=42)


# model.add(Flatten(input_shape=(max_size, max_size)))
# model.add(Dense(1000))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Flatten(input_shape=(max_size, max_size)))
# model.add(Dense(64))
model.add(Input(shape=(max_size, max_size)))
model.add(Flatten(input_shape=(max_size, max_size)))
model.add(Dense(factorial(8)))
model.add(Dense(1, activation='linear'))

adam = Adam()
model.compile(optimizer=adam,
              loss='mean_squared_error',
              metrics=['accuracy'],
              )

# Обучаем модель
model.fit(X_train, y_train, epochs=100)

# Оцениваем модель
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)
