import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.model_selection import cross_validate

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import metrics

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)


"""
Using Pandas Dataframe to make the data easily accessible. The data from reviews.csv gets split up into positive and negative reviews.
"""
def read_data():
	data = pd.read_csv('mainreviews.csv', sep = '|')
	data["Sentiment"] = ["Positive" if x in [4, 5] else "Negative" for x in data["Stars"]]
	data["Review"] = data['Review'].str.replace('[^\w\s]','') # Removing punctuation
	return data

"""
This function returns a tokenized version of the text input. The word_tokenize function comes from NLTK
"""
def tokenize(text):
	return word_tokenize(text)

def main():
	data = read_data() # Returns the data from read_data()
	dutch_stopwords = stopwords.words('dutch') # List of dutch stopwords to use in the CountVectorizer

	text_regression = Pipeline([('count', CountVectorizer(ngram_range=(1, 2), tokenizer = tokenize, stop_words = dutch_stopwords, analyzer = 'word')), ('tfidf', TfidfTransformer()), ('cla', LinearSVC())]) # Using a pipeline to combine the CountVecotorizer, the TfidfTransformer and the LogisticRegression classifier

	X_train, X_test, y_train, y_test = train_test_split(data, data.Sentiment, test_size=0.2, shuffle = True) # Splitting the data in 20% test data and 80% training data. The data is shuffled so it won't train on the same test data
	model = text_regression.fit(X_train.Review, y_train) # Fitting the model to the data
	predicted = model.predict(X_test.Review) # Predicting the test data

	cv_results = cross_validate(model, X_train.Review, y_train, cv = 10) # Running a 10-fold cross validation on the data
	all_results = cv_results['test_score'] # A list of results from the K-fold cross validation

	print ("Evaluation..")
	print()
	print("Confusion Matrix:")
	print(confusion_matrix(y_test, predicted))
	print()
	print("Precision, Recall and F-score table:")
	print(metrics.classification_report(y_test, predicted))
	print()

	print("Average accuracy:")
	print(sum(all_results)/10) # The average accuracy of all scores in the list


if __name__ == '__main__':
    main()
