#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import bigrams,trigrams
from nltk import pos_tag
from nltk import word_tokenize
import string

stop = set(stopwords.words('english'))
punctuation = set(string.punctuation + '–')
numbers = set(string.digits)
lemma = WordNetLemmatizer()

def tokenize(sentence):
    return word_tokenize(sentence)

def lower(doc):
    return [i.lower() for i in doc]

def get_pos_tags(doc):
    return pos_tag(doc)

def filter_pos_tags(doc, tags=('NN', 'VB', 'JJ', 'RB')):
    return [i[0] for i in get_pos_tags(doc) if i[1].startswith(tags)]

def get_bigrams(doc):
    bigram_list = []
    for bigram in bigrams(doc):
        bigram_list.append('_'.join(bigram))
    return bigram_list

def get_trigrams(doc):
    trigram_list = []
    for trigram in trigrams(doc):
        trigram_list.append('_'.join(trigram))
    return trigram_list

def remove_common_stopwords(doc):
    return [i for i in doc if i not in stop]

def remove_custom_stopwords(doc):
    with open('data/raw/custom_stopwords.txt', 'r') as f:
        custom_stopwords = f.read().splitlines()
        return [i for i in doc if i not in custom_stopwords]

def remove_punctuation(doc):
    return [token.translate(dict((ord(char), None) for char in punctuation)) for token in doc]

def remove_numbers(doc):
    return [token.translate(dict((ord(char), None) for char in numbers)) for token in doc]

def filter_pos(doc, filter):
    return [pos for pos in doc if pos in filter]

def filter_n_grams(doc, filter):
    return [ngram for ngram in doc if set(ngram.split('_')) < set(filter)]

def lemmatize(doc):
    return [lemma.lemmatize(word) for word in doc]

def validate_markers(sentence):
    sentence = sentence.split('CITATION')
    for index, token in enumerate(sentence):
        sentence[index] = token.strip()
    sentence = ' CITATION '.join(sentence)
    sentence = sentence.split('REFERENCE')
    for index, token in enumerate(sentence):
        sentence[index] = token.strip()
    sentence = ' REFERENCE '.join(sentence)
    sentence = sentence.replace('( CITATION', 'CITATION')
    sentence = sentence.replace('CITATION )', 'CITATION')
    sentence = sentence.replace('( REFERENCE', 'REFERENCE')
    sentence = sentence.replace('REFERENCE )', 'REFERENCE')
    sentence = ' '.join(sentence.split())
    return(sentence)

def remove_markers(sentence):
    sentence = validate_markers(sentence)
    sentence = sentence.replace('CITATION', '')
    sentence = sentence.replace('REFERENCE', '')
    sentence = ' '.join(sentence.split())
    return sentence

def clean_doc(doc):
    return [i for i in doc if i]
