# Spam Detection
# The dataset is an SMS Spam Collection is a set of SMS tagged messages and contains a set of SMS messages in Engish of 5574 messages 
# NLTK, Logistic regression, Naïve Bayes

# Importing libraries
import pandas as pd

# Importing dataset
df = pd.read_csv("SMS-SPAM-BAYES.csv")

# Preview data
df.head()

# Dataset dimensions - (rows, columns)
df.shape

# List of features
list(df)

df['type']
df['text']

# Text pre-processing
df['type'] = df.type.map({'ham': 0, 'spam': 1})

df['text'] = df.text.map(lambda x: x.lower())
df['text']

df['text'] = df.text.str.replace('[^\w\s]', '')
df['text']

import nltk
nltk.download()

# Tokenization
df['text'] = df['text'].apply(nltk.word_tokenize)
df['text']

# Stemming
from nltk.stem import PorterStemmer
PS = PorterStemmer()

df['text'] = df['text'].apply(lambda x: [PS.stem(y) for y in x])
df['text'] 

# Features extraction
# Count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
df['text'] = df['text'].apply(lambda x: ' '.join(x))
CV = CountVectorizer()
counts = CV.fit_transform(df['text'])

# 5574 x 8169 sparse matrix

from sklearn.feature_extraction.text import TfidfTransformer
trans = TfidfTransformer().fit(counts)
X = trans.transform(counts)
X.shape
type(X)

X1 = pd.DataFrame(X.todense())

# Train test splitting commands
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(counts, df['type'])

X_train.shape, X_test.shape, Y_train.shape,Y_test.shape

###################################################################################
# Supervised learning-Logistic regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train,Y_train)
y_pred = LR.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)

# Navie bayes learning
# multinomial NB
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)

Y_pred = MNB.predict(X_test)
Y_test

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_pred,Y_test)
acc

MNB.class_prior
MNB.class_count_
MNB.classes_
MNB.coef_
MNB.predict_proba
MNB.feature_count_

# Gaussian NB
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train, Y_train)

Y_pred1 = GNB.predict(X_test)
Y_test

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_pred1,Y_test)
acc1



