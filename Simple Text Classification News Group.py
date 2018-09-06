
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


# ## Load the Dataset
# Similar to our method in the MNIST tutorial we are going to be loading our data from sklearn's selection of datasets. Specifically the 20 news group dataset. Because we are using sklearn's datasets we can use functions of that import to immediately get training and validation data. 
# 
# To Do:
# - Assign both a training subset and validation subset


twenty_train = fetch_20newsgroups(subset = 'train', shuffle = True)
twenty_test = fetch_20newsgroups(subset = 'test')


twenty_train.target_names


twenty_train.target.shape


twenty_train.data[0]


# ## Extract the Features
# We will need to convert the text files in numerical feature vectors so we can perform machine learning. For this we will be specifically be using what is know as a bag of words. A bag of words breaks down the text file into the words that it is made of splitting them by space. The bag of words will store the amount of times that particular word occurs and assign the word a unique value. 
# 
# To Do:
# - Assign a CountVectorizer
# - Transform the training data.
# - Reduce the weights of common words using TF-IDF

count_vect = CountVectorizer()


x_train_count = count_vect.fit_transform(twenty_train.data)


x_train_count.data[:20]


tfidf_transformer = TfidfTransformer()


x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)


x_train_tfidf.data[:20]


x_test_count = count_vect.fit_transform(twenty_test.data)


x_test_tfidf = tfidf_transformer.fit_transform(x_test_count)


x_test_tfidf.data[:20]


clf = MultinomialNB().fit(x_train_tfidf, twenty_train.target)
clf.score(x_test_tfidf, twenty_test.target)


# ## Train and Predict with Naive Bayes
# We are going to train a Naive Bayes (NB) model to predict what category are articles fall under. 
# 
# To Do:
# - Create a pipeline object which uses NB
# - Train the model
# - Predict using the test data we loaded in

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB())])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predictions = text_clf.predict(twenty_test.data)

np.mean(predictions == twenty_test.target)


# ## Train and Predict with Support Vector Machines
# A Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression. It is most commonly used for classification problems. 
# 
# To Do:
# - Create another pipeline object which uses a SVM (SGDClassifier)
# - Train the model
# - Predict using the test data we loaded in

new_pipeline = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier())])

new_pipeline = new_pipeline.fit(twenty_train.data, twenty_train.target)
predictions = new_pipeline.predict(twenty_test.data)

np.mean(predictions == twenty_test.target)


# ## Tune Performance
# To increase the performance of our models we can use a grid search tool also apart of sklearn which lets us fine tune our models with specific parameters. 
# 
# To Do:
# - Define the parameters you want to use.
# - Assign the the grid search with the model you want to use and the parameters. 
# - Train the model.

text_tune = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier())])
					

parameters = {'vect__ngram_range' : [(1, 1), (1, 2)],
             'tfidf__use_idf' : (True, False),
             'clf__alpha' : (1e-2, 1e-3)}
			 
gs_clf = GridSearchCV(text_clf, parameters, n_jobs = 1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
			 
print("Best score:   {}".format(gs_clf.best_score_))
print("Parameters used:   {}".format(gs_clf.best_params_))

# ## Exercise
# 1. Use grid search for our second model and print the results

gs_clf = GridSearchCV(new_pipeline, parameters, n_jobs = 1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

print("Best score:   {}".format(gs_clf.best_score_))
print("Parameters used:   {}".format(gs_clf.best_params_))
