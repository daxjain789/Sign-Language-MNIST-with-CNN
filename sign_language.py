import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

print(tf.__version__)

train = pd.read_csv(r"G:\Coursera Course\Tensorflow-practice\My_practice\dataset\sign_mnist_train.csv")
test = pd.read_csv(r"G:\Coursera Course\Tensorflow-practice\My_practice\dataset\sign_mnist_test.csv")
print(train.head())

train_labels = train['label']
test_labels = test['label']
del train['label']
del test['label']

train_images = train.values
test_images = test.values

print(train_images.shape, test_images.shape)

# Normalize the data
x_train = train_images / 255.0
x_test = test_images / 255.0

# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print("image shape-", x_train.shape, x_test.shape)
print("label shape-", y_train.shape, y_test.shape)

model = Sequential()
model.add(layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(25, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data = (x_test, y_test), batch_size=128)

model.evaluate(x_test, y_test)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

# graph
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')