import csv
import cv2
import numpy as np
import sklearn

# Read data from directory
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) # skip header
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)



# A helper function: given a file path and direction, return image and angle
def get_data(source_path, direction, measurement):
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    if direction == 'left':
        measurement += 0.2
    elif direction == 'right':
        measurement -= 0.2
    return image, measurement



# Use a generator to build dataset on the fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i, direction in zip([0, 1, 2], ['center', 'left', 'right']):
                    source_path = batch_sample[i]
                    measurement = float(batch_sample[3])
                    image, measurement = get_data(source_path, direction, measurement)
                    images.append(image)
                    measurements.append(measurement)

                	# Flip images and steering measurements to augment the data
                    image_flipped = np.fliplr(image)
                    measurement_flipped = -measurement
                    if measurement > 0.2 or measurement < -0.2:
                        images.append(image)
                        images.append(image_flipped)
                        measurements.append(measurement)
                        measurements.append(measurement_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, )
validation_generator = generator(validation_samples, batch_size=32)


# The model is based on NVIDIA Architecture
from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# A helper function to build conv layer
def conv_layer(model, conv_outputs, kernel_size, input_shape):
    if input_shape:
        model.add(Convolution2D(conv_outputs, kernel_size, kernel_size, input_shape=input_shape))
    else:
        model.add(Convolution2D(conv_outputs, kernel_size, kernel_size))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))

model = Sequential()
# lambda layer to normalize image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#Cropping Images, see https://goo.gl/hArVfs
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))

conv_layer(model, 24, 5, input_shape=(90, 320, 3))
conv_layer(model, 36, 5, None)
conv_layer(model, 48, 5, None)
conv_layer(model, 64, 3, None)
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Dropout(1))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/32,\
                    validation_data=validation_generator, \
                    validation_steps=len(validation_samples)/32, nb_epoch=3)

model.save('model.h5')
