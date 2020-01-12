# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:16:19 2020

@author: Izabela Roman
"""

import pandas as pd

# read data
reviews = pd.read_excel ('corpus.xlsx') 
print (reviews)
       
# return the wordnet object value corresponding to the POS tag
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
import nltk
nltk.download('stopwords')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def clean_text(text):
    #lower text
    text = text.lower()
    # tokenize text and remove punctuation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if not x in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    #remove special characters
    text = [re.sub('\W+',' ', word ) for word in text]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
reviews["Review_clean"] = reviews["Review"].apply(lambda x: clean_text(x))

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()


reviews['neg'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['neg'])
reviews['neu'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['neu'])
reviews['pos'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['pos'])
reviews['compound'] = reviews['Review_clean'].apply(lambda x:sia.polarity_scores(x)['compound'])

#We will consider posts with a compound value greater than 0.2 as positive and less than -0.2 as negative.
#0.2;-0.2 -> 0
reviews['label'] = 0
reviews.loc[reviews['compound'] > 0.2, 'label'] = 1
reviews.loc[reviews['compound'] < -0.2, 'label'] = -1
reviews.head()
#Analiza descriptivă pentru scorurile comentariilor 
reviews['compound'].describe()

#create columns Word Count, Special Char Count
reviews['Word Count'] = [len(review.split()) for review in reviews['Review_clean']]

#analiza descriptiva numar cuvinte                         
reviews['Word Count'].describe()

#numarul total de cuvinte procesate
Total = reviews['Word Count'].sum()
print (Total)

#Now let's check how many total positives and negatives we have in this dataset:
print(reviews.label.value_counts())
#procent
print(reviews.label.value_counts(normalize=True) * 100)

#bar chart
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 8))

counts = reviews.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(reviews['Review_clean'])



#Split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, reviews['label'], test_size=0.25, random_state=1)


#Naive Bayes

from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

# regresia logistica

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='ovr',solver='lbfgs')
model = lr.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print ("RL Accuracy is {}".format(accuracy))

# random forest

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier as rfc
classifier = rfc()
classifier.fit(X_train,y_train)
print("RF Accuracy:",classifier.score(X_test,y_test))

# decision tree

from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
model1 = dct.fit(X_train,y_train)
print("DT Accuracy:",dct.score(X_test,y_test))


#testare comentariu nou cu modelul de regresie logistica
test = "I was really excited for this book and it just really fell flat for me. The big mystery was so incredibly obvious and so it was annoying to watch Sigourney make the wrong choices. I really struggled at the beginning of this book with all the islands and how everything related to one another. Part of the issue is all the island names sound so similar because it's literally (Name) Helle. It's not immediately obvious that they're named after the family that runs the island. So I kept reading about "
import numpy as np
x = cv.transform(np.array([test]))
proba = model.predict_proba(x)
classes=model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)
print("Comentariul va fi incadrat in urmatoarea categorie",resultdf)

#exportarea bazei de date ce va fi utilizata mai departe pentru analize
reviews.to_excel('Goodreads_review.xlsx', index=False)
