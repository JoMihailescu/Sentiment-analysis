# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:54:12 2019

@author: Ioana Mihailescu
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

def clean_text(text):
    #lower text
    text = text.lower()
    # tokenize text and remove puncutation
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


#create columns Word Count, Uppercase Char Count si Special Char Count
reviews['Word Count'] = [len(review.split()) for review in reviews['Review_clean']]

reviews['Uppercase Char Count'] = [sum(char.isupper() for char in review) \
                              for review in reviews['Review_clean']]                           

reviews['Special Char Count'] = [sum(char in string.punctuation for char in review) \
                            for review in reviews['Review_clean']] 

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
    text_counts, reviews['label'], test_size=0.3, random_state=1)


#Model Building and Evaluation
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

#exportarea bazei de date ce va fi utilizata mai departe pentru analize
reviews.to_excel('Goodreads_review.xlsx', index=False)