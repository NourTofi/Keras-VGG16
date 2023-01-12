import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

img = cv2.imread("dog.jpg")
img = cv2.resize(img, (224, 224))

model = keras.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dense(4096, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.build()
model.summary()

result = model.predict(np.array([img]))
print(result)




