import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import ssl

#fixing ssl for getting dataset
ssl._create_default_https_context = ssl._create_unverified_context

#get dataset

(training_images, trainning_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

# class options

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#Reducing training images(This is optionel)

# training_images = training_images[:1000000]
# trainning_labels = trainning_labels[:1000000]
# testing_images = testing_images[:200000]
# testing_labels = testing_labels[:200000]

# model

model = models.Sequential()

# neural network
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

# Dense
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

# Compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training
model.fit(training_images, trainning_labels, epochs=10, validation_data=(testing_images, testing_labels))

#Ending
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')
