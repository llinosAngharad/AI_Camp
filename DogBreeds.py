# coding: utf-8

# In[1]:


# Llinos
import pandas as pd         #read in csv data
import numpy as np
# graphical representation of progress
import tqdm as tqdm         
import cv2 as cv            # Open images
import matplotlib.pyplot as plt    # Plot graphs
import matplotlib.image as mpimg

from keras.layers import Activation, Convolution2D, Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.applications import InceptionV3, ResNet50, Xception
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from numpy.random import seed

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[2]:


get_ipython().magic(u'matplotlib inline')
seed(seed=1982)


# ## Find the Data and Load
# Using console commands from inside jupyter, we can find the files we need. 
# 
# To Do:
# - Find the data we need and load them into the dataframe

# In[3]:


PATH = 'data/'
train_df = pd.read_csv(PATH+'labels.csv')
test_df = pd.read_csv(PATH+'sample_submission.csv')


# In[4]:


train_df.head(25)


# ## Reduce the Amount of Data
# Since there is a lot of images to work with, we are going to have to reduce the amount that we will be using so we can get faster results and also train with more even spread of classes. 
# 
# To Do:
# - Define the desired image height and width for every image we pass in.
# - Define the amount of breeds you want to work with and reduce the dataframe to only those breeds.
# - Use one hot encoding to get a unique identifier for each breed.

# In[5]:


IMG_HEIGHT = 250
IMG_WIDTH = 250
NUM_CLASSES = 16

images = []
classes = []


# In[6]:


selected_breeds = list(train_df.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)


# In[7]:


df_sub_train = train_df[train_df['breed'].isin(selected_breeds)]


# In[8]:


targets = pd.Series(df_sub_train['breed'])
one_hot = pd.get_dummies(targets, sparse = True)
one_hot_labels = np.asarray(one_hot)


# In[9]:


selected_breeds


# In[10]:


img = mpimg.imread(PATH+'train/{}.jpg'.format(df_sub_train.iloc[3]['id']))
plt.title('Doggo')
plt.imshow(img)


# In[11]:


one_hot.head()


# In[12]:


print("Dimensions of one_hot_labels: {}".format(one_hot_labels.shape))
one_hot_labels


# ## Load the Images and Labels into Arrays
# We need a way to access each image in the order that they are in the table. Create a for loop and load each image in using the table with each image name associated with its breed and ID.
# 
# To Do:
# - Create a for loop that loads in each image 
# - Assign the labels to an array. 

# In[13]:


for f, breeds in tqdm.tqdm(df_sub_train.values):
    img = cv.imread(PATH+'train/{}.jpg'.format(f))
    images.append(cv.resize(img, (IMG_HEIGHT, IMG_WIDTH)))
    
classes = one_hot_labels

print("classes size: {}".format(classes.shape))
print(classes[:5])


# ## Split the Data
# Split the data into training and validation data.
# 
# To Do:
# - Split the data using train_test_split

# In[14]:


X = np.array(images, dtype=np.float32)
Y = np.array(classes, dtype=np.uint8)

print(X.shape)
print(Y.shape)


# In[15]:


x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, shuffle=True)


# In[16]:


print("X training shape: {}".format(x_train.shape))
print("Y training shape: {}".format(y_train.shape))
print("X validation shape: {}".format(x_val.shape))
print("Y validation shape: {}".format(y_val.shape))


# ## Build the Model
# This time we are going to use a pre-trained model and build our up our own layers on top of it. This allows us to use a model which is already pretty accurate by itself and then train new weights that are learned from our particular dataset. 
# 
# To Do:
# - Define an InceptionV3 model 
# - Build the following layers on top of the pre-trained model's outputs: 
#         Flatten
#         Dense
#         Droput
#         Dense
# - Define a copy of the model with an extra Dense layer which will provide us with the predictions
# - Compile the model
# - Augment the data now or after to see the difference in results
# - Train the model with your training data, validation data and a set amount of epochs based on the machine your are running. 

# In[17]:


starting_model = InceptionV3(include_top=False,
                            weights='imagenet',
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                            classes=NUM_CLASSES)


# In[18]:


model = starting_model.output
model = Flatten()(model)
model = Dense(1024, activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(1024, activation='relu')(model)

predictions = Dense(NUM_CLASSES, activation='softmax')(model)

final_model = Model(inputs=[starting_model.input], outputs=[predictions])


# In[19]:


final_model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[20]:


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                  rotation=)
valid_datagen = ImageDataGenerator(rescale=1. /255)


train_datagen_2 = ImageDataGenerator(rotation_range=40, rescale=1. / 255)
valid_datagen_2 = ImageDataGenerator(rescale=1. /255)


# In[32]:


train_generator = train_datagen.flow(x_train, y_train, batch_size=20)

val_generator = valid_datagen.flow(x_test, y_test, batch_size=20)

# Augmented image generators
train_generator_2 = train_datagen_2.flow_from_directory(PATH+'train',
                                                   target_size=(250, 250),
                                                   batch_size=16,
                                                   class_mode='binary')

val_generator_2 = valid_datagen_2.flow_from_directory(PATH+'test',
                                                   target_size=(250, 250),
                                                   batch_size=16,
                                                   class_mode='binary')


# In[ ]:


final_model.fit_generator(train_generator,
         steps_per_epoch=2000/ 16,
         epochs=5,
         validation_data = val_generator,
         validation_steps = 800 / 16)


# ## Exercises
# 
# 1. Build another model but this time with a different pre-trained model and your own layers.
# 2. Using just a pre-trained model with no additional layers, perform logistic regression on the predictions and look at the accuracy. 
