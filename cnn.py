# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # help to initial the ANN as a sequence of layers
from keras.layers import Conv2D # 1st step of making the CNN, that is the convolution step in whic we add the convolutional layers
from keras.layers import MaxPooling2D # 2st step, the pooling step - pooling layers
from keras.layers import Flatten # 3st step flattening, in which we convert all the pooled feature maps that we have created through
                                # convolution and max pooling into this large feature vector, that is then becoming the input of our fully connected layers 
from keras.layers import Dense # to add fully connecting layers into classic ANN

# Initialising the CNN
classifier = Sequential() # creating an object

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# 32 - number of feature detectors, 32 is common practise, then 64, 128,256
# 3,3 - number of rows and columns in the feature detector
# input_shape , we have to convert all images to the same format, first 3 stands for colour
# activation function, to make sure we don't have any negative pixel value in feature maps 
#                      in order to have nonlinearity in our convolutional neural network

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# to reduce the feature map and reduce the number of notes in fully connected convolutional neural networks
# with stride 2 so the pooled feature map will half of the input image

# Adding a second convolutional layer    -> # improved results, reduced overfitting
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# don't have to include input_shape because we are getting them from the first layer

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection (adding the hidden and output layer)
classifier.add(Dense(units = 128, activation = 'relu'))

# units = number of nodes in the hidden layer, parameter to tune, common practise to choose power of two 
# rectifiyer activation function for hidden layer

classifier.add(Dense(units = 1, activation = 'sigmoid'))
# units = 1 output layer
# sigmoid because of two categories, otherwise softmax

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images # important to not overfit

from keras.preprocessing.image import ImageDataGenerator

# random transformations to produce more images to train on
# code from keras.io/preprocessingimage/ 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2, 
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), #dimensions specified in convolution
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64), # to get better results, enhance the sizes, so we work with more pixels
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, #number of images in the training set
                         epochs = 25, # 50 was too many
                         validation_data = test_set,
                         validation_steps = 2000)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) # because we have the third dimension (3 - colourful picture)
test_image = np.expand_dims(test_image, axis = 0) #adding another dimension representing the batch
# axis specifies the position of the dimension we are adding
result = classifier.predict(test_image)
training_set.class_indices # to find out wheter 1 corresponds to dog or cat
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) # because we have the third dimension (3 - colourful picture)
test_image = np.expand_dims(test_image, axis = 0) #adding another dimension representing the batch
# axis specifies the position of the dimension we are adding
result = classifier.predict(test_image)
training_set.class_indices # to find out wheter 1 corresponds to dog or cat
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


