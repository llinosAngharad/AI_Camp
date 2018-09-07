import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import pickle
import seaborn as sns


# ## Load and Clean the Data
# Load the data as before with pandas and take a look at the data so we know what we will be working with. 
# 
# To Do:
# - Load the dataset into a dataframe with pandas
# - Clean the data so it is easier to work with.

tweets = pd.read_csv('train.csv')
tweets.head(5)

import re

tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: x.lower())
tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x: 
                                                      re.sub('[^a-z0-9\s]', '', x))
tweets.head()

sns.countplot(x = 'Sentiment', data = tweets)


# ## Tokenise
# We are going to use tokenisation to find the 2000 most common words. Tokenising is used on a sequence of words (basically a sentence) to count their frequency. A *token* is an instance of a sequence of characters (basically a word). A type is the class of all tokens containing the same character sequence.
# 
# To Do:
# - Define the max features
# - Initialise a tokeniser with the max feature and a defined split
# - Calculate the different word frequencies 
# - Add padding to the sequences

max_features = 2000

tokenizer = Tokenizer(num_words = max_features, split = ' ')
tokenizer.fit_on_texts(tweets['SentimentText'].values)

tweet_data = tokenizer.texts_to_sequences(tweets['SentimentText'].values)


print tweet_data[0]
print tweet_data[12]

from keras.preprocessing.sequence import pad_sequences

tweet_data = pad_sequences(tweet_data)
print tweet_data[0]
print tweet_data[12]


# ## Build the Model 
# As before define and build the model you want to train with this dataset. 
# 
# To Do:
# - Define a sequential model and add your layers. 
# - Compile the model
# - (Optional) Display the layers with summary()

embed_dim = 128
lstm_out = 128

model = Sequential()

model.add(Embedding(max_features, embed_dim, input_length = tweet_data.shape[1]))
model.add(Dropout(0.5))

model.add(Conv1D(64, 5, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 4))
model.add(Dropout(0.6))

model.add(LSTM(lstm_out, dropout = 0.4, recurrent_dropout = 0.4))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
             metrics = ['accuracy'])



print(model.summary())


# ## Split the Data and Train
# Again we need to split the data so we have testing data and validation data to work with. The easiest way to do this is to make use of the scikit-learn library's function train_test_split.
# 
# To Do:
# - Train the model with the data you have just organised

y = pd.get_dummies(tweets['Sentiment']).values
print y


x_train, x_test, y_train, y_test = train_test_split(tweet_data, y, test_size = 0.3, random_state = 42)

partial_x_train = x_train[:5000]
partial_y_train = y_train[:5000]

partial_x_test = x_test[:1000]
partial_y_test = y_test[:1000]

batch_size = 500
model.fit(partial_x_train, 
          partial_y_train,
          epochs = 10,
          batch_size = batch_size,
          validation_data = (partial_x_test, partial_y_test))


model.save("tweet_sentiment_model.hdf5")


# ## Exercises
# 
# 1. Using our trained model make a prediction against the test data.
# 

model = load_model("tweet_sentiment_model.hdf5")


test_tweets = pd.read_csv('train.csv')
test_tweets.head(10)

test_tweets['SentimentText'] = test_tweets['SentimentText'].apply(lambda x: x.lower())
test_tweets['SentimentText'] = test_tweets['SentimentText'].apply(lambda x: 
                                                      re.sub('[^a-z0-9\s]', '', x))

tokenizer.fit_on_texts(test_tweets['SentimentText'].values)
tweet_data = tokenizer.texts_to_sequences(test_tweets['SentimentText'].values)


tweet_data = pad_sequences(tweet_data)
y = test_tweets['Sentiment'].values


predictions = model.predict_classes(tweet_data)


from sklearn.metrics import classification_report
print( classification_report(y[:1000], predictions[:1000]))


# 2. Build another model and try and achieve a higher accuracy than us.
