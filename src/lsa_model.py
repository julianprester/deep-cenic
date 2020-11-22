#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import logging
import gensim
import pickle
import preprocessing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_dir = 'models/lsa/'

def generate_dictionary(documents):
    return gensim.corpora.Dictionary(documents)

def generate_corpus(documents, dictionary):
    return [dictionary.doc2bow(doc) for doc in documents]

def generate_tfidf(corpus):
    return gensim.models.TfidfModel(corpus)

def generate_lsa_model(dictionary, corpus, tfidf):
    return gensim.models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=300)

def preprocess_corpus(documents):
    documents = list(map(preprocessing.tokenize, documents))
    documents = [preprocessing.remove_punctuation(doc) for doc in documents]
    documents = [preprocessing.remove_numbers(doc) for doc in documents]
    documents = [preprocessing.lower(doc) for doc in documents]
    documents = [preprocessing.remove_common_stopwords(doc) for doc in documents]
    documents = [preprocessing.clean_doc(doc) for doc in documents]
    documents = [doc for doc in documents if doc]
    return documents

def generate_title_model():
    CP = pd.read_csv('data/interim/CP.csv')
    LR = pd.read_csv('data/interim/LR.csv')
    documents = []

    for index, row in CP.iterrows():
        if not pd.isnull(row['title']):
            documents.append(row['title'])
    for index, row in LR.iterrows():
        if not pd.isnull(row['title']):
            documents.append(row['title'])

    documents = preprocess_corpus(documents)
    dictionary = generate_dictionary(documents)
    corpus = generate_corpus(documents, dictionary)
    tfidf = generate_tfidf(corpus)
    lsa_model = generate_lsa_model(dictionary, corpus, tfidf)

    dictionary.save(model_dir + 'title.dict')
    gensim.corpora.MmCorpus.serialize(model_dir + 'title.mm', corpus)
    lsa_model.save(model_dir + 'title.model')
    with open(model_dir + 'title.docs', 'wb') as docs_file:
        pickle.dump(documents, docs_file, pickle.HIGHEST_PROTOCOL)

def generate_abstract_model():
    CP = pd.read_csv('data/interim/CP.csv')
    LR = pd.read_csv('data/interim/LR.csv')
    LR = LR[LR['abstract'].notnull()]
    LR = LR.loc[:,['citation_key_lr','abstract']]

    documents = []

    for index, row in CP.iterrows():
        if not pd.isnull(row['abstract']):
            documents.append(row['abstract'])
    for index, row in LR.iterrows():
        if not pd.isnull(row['abstract']):
            documents.append(row['abstract'])

    documents = preprocess_corpus(documents)
    dictionary = generate_dictionary(documents)
    corpus = generate_corpus(documents, dictionary)
    tfidf = generate_tfidf(corpus)
    lsa_model = generate_lsa_model(dictionary, corpus, tfidf)

    dictionary.save(model_dir + 'abstract.dict')
    gensim.corpora.MmCorpus.serialize(model_dir + 'abstract.mm', corpus)
    lsa_model.save(model_dir + 'abstract.model')
    with open(model_dir + 'abstract.docs', 'wb') as docs_file:
        pickle.dump(documents, docs_file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    generate_title_model()
    generate_abstract_model()
