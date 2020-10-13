# this uses Non-negative Matrix Factorization to extract topics from a corpus.
# follows tutorial https://medium.com/@obianuju.c.okafor/automatic-topic-classification-of-research-papers-using-the-nlp-topic-model-nmf-d4365987ec82
import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import re
import gensim
import gensim.corpora as corpora
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
stemmer = SnowballStemmer('english')

f = open('./biden_speeches.txt', 'r')
documents = f.read()
# function for lemmatization
def get_lemma(word):
    return stemmer.stem(WordNetLemmatizer().lemmatize(word, pos='v'))

#tokenization
tokenized_data = documents.split(' ')
temp_tokenized_data = []
# remove punctuation
for item in tokenized_data:
    item = re.sub('[-()\\"\'!?\n]', '', item)
    temp_tokenized_data.append(item)
tokenized_data = []
for item in temp_tokenized_data:
    item = re.sub('[.,]', ' ', item)
    tokenized_data.append(item)

# remove stop words
stop_words = stopwords.words('english')
stop_words.extend(['folks', 'fact', 'look', 'thank you'])

## remove words of length less than 3
temp_tokenized_data = []
for item in tokenized_data:
    if item not in stop_words and len(item) > 3:
        temp_tokenized_data.append(item)

tokenized_data = []
# lemmatize by calling lemmatization function
for item in temp_tokenized_data:
    tokenized_data.append(get_lemma(item))

print(tokenized_data)