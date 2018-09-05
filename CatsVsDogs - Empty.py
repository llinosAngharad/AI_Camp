import pandas as pd 
import numpy as np
import matplotlib as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# ## Define Paths to the Image Data
# As we are working with images and not a dataset we only need to define the paths to the images. When working with just images we sometimes need to split the images further into seperate folders which will be defined as classes or labels later on.
# 
# To Do:
# - Find the correct paths to the training set and testing set
# - Define two seperate paths to the images within these folders
get_ipython().system(u'ls')
PATH = 'data/'
get_ipython().system(u'ls {PATH}')


# ## Data Augmentation Example
# Here we are gonna augent one of our images and put it in a folder called preview. This stage is to only show you how to go about augmenting your data with keras. Later we will use to hopefully improve our model. 
# 
# To Do:
# - Define an ImageDataGenerater with different parameters
# - Find an image that you want to augment and convert to an array
# - Use a for loop and your generator to generate the augmented images
datagen = ImageDataGenerator(rotation_range = 40, rescale = 1. / 255, horizontal_flip = True)

img = load_img(PATH + 'Train/cats/0.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1, save_to_dir = PATH + 'preview', save_format = 'jpeg'):
    i += 1
    if i > 20:
        break


# ## Augment the Images
# Now that we have seen what the accuracy of our model before adding any extra data we will now augment the images to provide extra variations of our images. This will also help generalise our model so it doesn't overfit on our training data. We only want to add these variations to the training data and not the validation. 
# 
# To Do:
# - Build an image generator like we did previously but for both the training data and validation data
# - Build a training generator and a validation generator using the images from the image generators we made
# - Train the same model again with the new images
train_datagen = ImageDataGenerator(rescale = 1. / 255)
valid_datagen = ImageDataGenerator(rescale = 1. / 255)

train_datagen_2 = ImageDataGenerator(rotation_range = 40, rescale = 1. / 255, horizontal_flip = True)
valid_datagen_2 = ImageDataGenerator(rescale = 1. / 255)

train_datagenerator = train_datagen.flow_from_directory(PATH + 'Train', 
                                                       target_size = (150, 150),
                                                       batch_size = 16,
                                                       class_mode = 'binary')

valid_datagenerator = train_datagen.flow_from_directory(PATH + 'Test', 
                                                       target_size = (150, 150),
                                                       batch_size = 16,
                                                       class_mode = 'binary')

train_generator_2 = train_datagen_2.flow_from_directory(PATH + 'Train', 
                                                       target_size = (150, 150),
                                                       batch_size = 16,
                                                       class_mode = 'binary')

valid_generator_2 = valid_datagen_2.flow_from_directory(PATH + 'Train', 
                                                       target_size = (150, 150),
                                                       batch_size = 16,
                                                       class_mode = 'binary')


# ## Build the Model
# With a set of images very little work has to be done before the model is created and trained. However, when we model first the likelihood of getting a high accuracy is low unless we make use of a pre-trained model with specified weights. This is not a bad thing though!
# 
# To Do: 
# - Build a Sequential model with multiple layers
# - Compile the built model
# - Train the model
data_format = 'channels_first'

model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape = (150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(data_format = data_format, pool_size = (2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(data_format = data_format, pool_size = (2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(data_format = data_format, pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit_generator(train_generator_2, 
          steps_per_epoch = 2000 / 16, 
          epochs = 50, 
          validation_data = valid_generator_2,
          validation_steps = 800 / 16)


# ## Exercise 
# 1. Try and improve the model above the accuracy you are getting at 5 epochs. Things to try:
#         - Add more layers
#         - Change the number of epochs
#         - Augment your data more. 
