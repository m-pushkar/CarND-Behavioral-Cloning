import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D
import matplotlib.pyplot as plt

# List for storing the data
data = [] 

# Read log file of the recorded data
with open('./data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    # Skip header row
    next(reader, None)
    for line in reader:
        data.append(line) 

# Split data set for training and validation
train_sample, validation_sample = train_test_split(data, test_size = 0.20)

# Generator function
def generator(data, batch_size = 32):
    num_samples = len(data)
    while 1: 
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            samples = data[offset: offset + batch_size]

            images = []
            measurements = []
            # Correction parameter for side cameras
            correction = 0.2
            for sample in samples:
                    for i in range(0,3): # 3 Camera images          
                        name = './data/IMG/' + sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) # BGR to RGB conversion
                        measurement = float(sample[3]) # Steering angle measurement
                        images.append(center_image) 
                        images.append(cv2.flip(center_image, 1)) # Image flipping

                        # Taking the steering measurements
                        # i=0 --> Center camera image
                        # i=1 --> Left camera image
                        # i=2 --> Right camera image

                        # Steering angle measurement correction
                        if(i==0):
                            measurements.append(measurement)
                        elif(i==1):
                            measurements.append(measurement + correction) 
                        elif(i==2):
                            measurements.append(measurement - correction)
                        
                        # Steering angle measurement correction for flipped image
                        if(i==0):
                            measurements.append(-(measurement))
                        elif(i==1):
                            measurements.append(-(measurement + correction))
                        elif(i==2):
                            measurements.append(-(measurement - correction)) 

            # Numpy arrays
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            # Yield generator
            yield sklearn.utils.shuffle(X_train, y_train)

training_generator = generator(train_sample, batch_size = 32)
validation_generator = generator(validation_sample, batch_size = 32)

# Neural Network
model = Sequential()
# Image normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
# Image cropping
model.add(Cropping2D(cropping = ((70,25),(0,0))))  
# Convolution feature map with 24 filters, 5*5 kernal and 2*2 stride
model.add(Conv2D(24, (5, 5), strides=(2, 2)))                       
# Exponential Linear Unit (elu) activation layer 
model.add(Activation('elu'))                                       
# Convolution feature map with 36 filters, 5*5 kernal and 2*2 stride
model.add(Conv2D(36, (5, 5), strides=(2, 2)))
# Exponential Linear Unit (elu) activation layer
model.add(Activation('elu'))                                       
# Convolution feature map with 48 filters, 5*5 kernal and 2*2 stride
model.add(Conv2D(48, (5, 5), strides=(2, 2)))                       
# Exponential Linear Unit (elu) activation layer
model.add(Activation('elu'))                                       
# Convolution feature map with 64 filters, 3*3 kernal and 1*1 stride
model.add(Conv2D(64, (3, 3)))                                       
# Exponential Linear Unit (elu) activation layer
model.add(Activation('elu'))                                       
# Convolution feature map with 64 filters, 3*3 kernal and 1*1 stride
model.add(Conv2D(64, (3, 3)))                                       
# Exponential Linear Unit (elu) activation layer
model.add(Activation('elu'))                                       
# Flattening layer
model.add(Flatten())                                               
# Fully connected layer         
model.add(Dense(100))                                               
# Exponential Linear Unit (elu) activation layer
model.add(Activation('elu'))                                       
# Dropout layer
model.add(Dropout(0.25))                                           
# Fully connected layer
model.add(Dense(50))                                               
# Exponential Linear Unit (elu) activation layer
model.add(Activation('elu'))                                       
# Fully connected layer
model.add(Dense(10))                                               
# Exponential Linear Unit (elu) activation layer
model.add(Activation('elu'))                                       
# Output
model.add(Dense(1))                                                 
# Mean squared error loss function to minize the loss and Adam optimizer
model.compile(loss='mse',optimizer='adam')     

history_object = model.fit_generator(training_generator, 
                steps_per_epoch = int(len(train_sample) / 32), 
                validation_data = validation_generator, 
                validation_steps = int(len(validation_sample) / 32),
                epochs=3)

# Save model
model.save('model.h5')

# Summary display
model.summary()

# Plot training and validation loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()