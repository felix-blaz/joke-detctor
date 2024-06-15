"""
Created on Wed Feb  7 11:42:55 2024

@author: aurelia power
"""

import re, pandas as pd, numpy as np, matplotlib.pyplot as plt
import nltk
from collections import Counter 
import warnings 
warnings.filterwarnings('ignore') 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import classification_report 
from sklearn.model_selection import cross_val_predict 
import wordcloud 

from sklearn.cluster import KMeans

#############################################
############ text visualisations ############

"""
takes in a list of tagged documents and a POS(as a string) 
returns the normalised count of a POS for each tagged document 
"""
def normalisePOSCounts(tagged_docs, pos):
    counts = [] 
    for doc in tagged_docs:
        count = 0 
        for pair in doc:
            if pair[1] == pos:
                count += 1 
        counts.append(count) 
    lengths = [len(doc) for doc in tagged_docs] 
    return [count/length for count, length in zip(counts, lengths)] 

"""
takes in a list of documents, a POS(as a string), and a list of categories/labels 
it tags the documents and calls the above function 
it then plots the normalised frequency of the POS across all labels 
"""
def plotPOSFreq(docs, pos, labels):
    tagged_docs = [nltk.pos_tag(nltk.word_tokenize(doc)) for doc in docs] 
    normalised_counts = normalisePOSCounts(tagged_docs, pos) 
    plt.bar(np.arange(len(docs)), normalised_counts, align='center') 
    plt.xticks(np.arange(len(docs)), labels, rotation=40) 
    plt.xlabel('Label (Category)') 
    plt.ylabel(pos + ' frequency') 
    plt.title('Frequency distribution of ' + pos) 

## function to generate the word cloud for a given topic/class
def generate_cloud(text, topic, bg_colour='black', min_font=10):
    cloud = wordcloud.WordCloud(width=700, height=700, random_state=1, background_color=bg_colour, min_font_size=min_font).generate(text) 
    plt.figure(figsize=(7, 7), facecolor=None) 
    plt.imshow(cloud) 
    ##plt.axis('off') 
    plt.tight_layout(pad=0) 
    plt.xlabel(topic) 
    plt.xticks([]) 
    plt.yticks([]) 

## function to generate multiple word clouds for a set of topics/classes/categories
def generate_wordclouds(texts, categories, bg_colour, min_font=10):
  fig = plt.figure(figsize=(21, 7))
  for i in range(len(texts)):
    ax = fig.add_subplot(1,3,i+1)
    cloud = wordcloud.WordCloud(width=700, height=700, random_state=1, 
                      background_color=bg_colour, 
                      min_font_size=min_font).generate(texts[i])
    ax.imshow(cloud)
    ax.axis('off')
    ax.set_title(categories[i])

#########################################################
########## vectorisation functions ######################
"""=minFont
NOTE: all vectorisers from sklearn discard punctuation, which may not be appropriate.
So, I have specified a regex to deal with this situation.
"""
token_regex = r"\w+(?:'\w+)?|[^\w\s]" 

"""
takes in a list of documents, applies the CountVectoriser from sklearn
using the following params by default: decode_error='replace', strip_accents=None, 
lowercase=False, ngram_range=(1, 1)  then it builds and returns a data frame 
"""
def build_count_matrix(docs, decode_error='replace', strip_accents=None, lowercase=False, token_pattern=token_regex, ngram_range=(1, 1)):    
    vectorizer = CountVectorizer(decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range) 
    X = vectorizer.fit_transform(docs) 
    terms = list(vectorizer.get_feature_names_out()) 
    count_matrix = pd.DataFrame(X.toarray(), columns=terms) 
    return count_matrix.fillna(0) 

## function to generate a matrix with normalised frequencies  same params as above
def build_tf_matrix(docs, decode_error='replace', strip_accents=None, lowercase=False, token_pattern=token_regex, ngram_range=(1, 1)):
    count_matrix = build_count_matrix(docs, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range) 
    doc_lengths = count_matrix.sum(axis=1) 
    return count_matrix.divide(doc_lengths, axis=0) 

## function to generate a matrix with tfidfs scores  same params as above
def build_tfidf_matrix(docs, decode_error='replace', strip_accents=None, lowercase=False, token_pattern=token_regex, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range) 
    X = vectorizer.fit_transform(docs) 
    terms = list(vectorizer.get_feature_names_out()) 
    tfidf_matrix = pd.DataFrame(X.toarray(), columns=terms) 
    return tfidf_matrix.fillna(0) 

#########################################################
############# validation functions ######################

"""function to train and x-validate across acc, rec, prec  
and get the classification report"""
def printClassifReport(clf, X, y, folds=5):
    predictions = cross_val_predict(clf, X, y, cv=folds) 
    print(classification_report(y, predictions)) 


#########################################################
############# word stats functions ######################
## function to print the n most frequent tokens in a text belonging to a given topic
def print_n_mostFrequent(topic, text, n):
    tokens = nltk.word_tokenize(text) 
    counter = Counter(tokens) 
    n_freq_tokens = counter.most_common(n) 
    print("=== "+ str(n) + " most frequent tokens in "  + topic + " ===") 
    for token in n_freq_tokens:
        print("\tFrequency of", "\"" + token[0] + "\" is:", token[1]/len(tokens)) 
        
## function to find the frequency of a token in several texts belonging to same topics/classes        
def token_percentage(token, texts):
    token_count = 0 
    all_tokens_count = 0 
    for text in texts:
        tokens = nltk.word_tokenize(text) 
        token_count += tokens.count(token) 
        all_tokens_count += len(tokens) 
    return token_count/all_tokens_count * 100 
        
#########################################################
############# preprocessing functions ###################
## function to carry out some initial cleaning
def clean_doc(doc, clean_operations):
    for key, value in clean_operations.items():
        doc = re.sub(key, value, doc) 
    return doc 

## function to resolve contractions
def resolve_contractions(doc, contr_dict):
    for key, value, in contr_dict.items():
        doc = re.sub(key, value, doc) 
    return doc 

########################################################################
############################# clustering ###############################
def k_means_clustering(X, k=2, initialisation='random'):
    model = KMeans(k, init=initialisation, random_state=1)
    model.fit(X)
    cluster_labels = model.labels_
    cluster_centers = model.cluster_centers_
    return (model, cluster_labels, cluster_centers)
