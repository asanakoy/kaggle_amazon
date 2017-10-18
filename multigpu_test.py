import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

from kerasext import make_parallel

model = Sequential()
model.add(Dense(4000, input_dim=8000, activation='tanh'))
model.add(Dense(2000, input_dim=8000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print (model.summary())

model = make_parallel(model, 2)
optimizer = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

x = np.random.rand(131072, 8000)
y = np.random.randint(0, 2, (131072, 1))

model.fit(x, y, batch_size=2048 * 4)