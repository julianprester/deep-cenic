#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import gensim
import os
import time
import pickle
import preprocessing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

lda_params = dict(
    model_dir = 'models/lda/',
    num_topics = 50,
    num_passes = 50,
    markers = True,
    tokenize = True,
    punctuation = True,
    numbers = True,
    common_stopwords = True,
    custom_stopwords = True,
    bigrams = True,
    trigrams = True,
    lemmatize = True,
    pos_tags = ('NN', 'VB')
)

def generate_dictionary(documents):
    return gensim.corpora.Dictionary(documents)

def generate_corpus(documents, dictionary):
    return [dictionary.doc2bow(doc) for doc in documents]

def generate_lda_model(corpus, dictionary, num_topics):
    lda = gensim.models.ldamulticore.LdaMulticore
    return lda(corpus, num_topics=num_topics, id2word = dictionary, batch=True, random_state=0, iterations=500, passes=lda_params['num_passes'])

def read_documents(context=True):
    CITATION = pd.read_csv('data/interim/CITATION.csv')
    CITATION = CITATION.dropna(subset=['citation_sentence'])
    CITATION = CITATION.fillna('')
    CITATION['citation_sentence'] = CITATION['citation_sentence'].astype(str)
    if context:
        CITATION['predecessor'] = CITATION['predecessor'].astype(str)
        CITATION['successor'] = CITATION['successor'].astype(str)
        CITATION['context'] = CITATION['predecessor'] + ' ' + CITATION['citation_sentence'] + ' ' + CITATION['successor']
        CITATION = CITATION.groupby(['citation_key_lr', 'citation_key_cp'])['context'].apply(lambda x: ' '.join(x)).reset_index()
        documents = CITATION['context'].dropna().tolist()
    else:
        CITATION = CITATION.groupby(['citation_key_lr', 'citation_key_cp'])['citation_sentence'].apply(lambda x: ' '.join(x)).reset_index()
        documents = CITATION['citation_sentence'].dropna().tolist()
    return documents

def build_model(documents):
    if lda_params['markers']:
        documents = map(preprocessing.remove_markers, documents)
    if lda_params['tokenize']:
        documents = map(preprocessing.tokenize, documents)
    documents = list(documents)
    if lda_params['pos_tags'] != ():
        tags = [preprocessing.lower(preprocessing.filter_pos_tags(doc, tags=lda_params['pos_tags'])) for doc in documents]
    if lda_params['punctuation']:
        documents = [preprocessing.remove_punctuation(doc) for doc in documents]
    if lda_params['numbers']:
        documents = [preprocessing.remove_numbers(doc) for doc in documents]
    documents = [preprocessing.lower(doc) for doc in documents]
    if lda_params['bigrams']:
        bigrams = [preprocessing.get_bigrams(doc) for doc in documents]
    if lda_params['trigrams']:
        trigrams = [preprocessing.get_trigrams(doc) for doc in documents]
    if lda_params['common_stopwords']:
        documents = [preprocessing.remove_common_stopwords(doc) for doc in documents]
    if lda_params['custom_stopwords']:
        documents = [preprocessing.remove_custom_stopwords(doc) for doc in documents]
    if lda_params['pos_tags'] != ():
        documents = [preprocessing.filter_pos(documents[i], tags[i]) for i in range(0, len(documents))]
    documents = [preprocessing.clean_doc(doc) for doc in documents]
    if lda_params['bigrams']:
        bigrams = [preprocessing.filter_n_grams(bigrams[i], documents[i]) for i in range(0, len(documents))]
    if lda_params['trigrams']:
        trigrams = [preprocessing.filter_n_grams(trigrams[i], documents[i]) for i in range(0, len(documents))]
    if lda_params['bigrams'] and not lda_params['trigrams']:
        documents = [documents[i] + bigrams[i] for i in range(0, len(documents))]
    if lda_params['trigrams'] and not lda_params['bigrams']:
        documents = [documents[i] + trigrams[i] for i in range(0, len(documents))]
    if lda_params['bigrams'] and lda_params['trigrams']:
        documents = [documents[i] + bigrams[i] + trigrams[i] for i in range(0, len(documents))]
    if lda_params['lemmatize']:
        documents = [preprocessing.lemmatize(doc) for doc in documents]
    documents = [preprocessing.clean_doc(doc) for doc in documents]
    documents = [doc for doc in documents if doc]
    
    dictionary = generate_dictionary(documents)
    corpus = generate_corpus(documents, dictionary)
    lda_model = generate_lda_model(corpus, dictionary, lda_params['num_topics'])

    if not os.path.exists(lda_params['model_dir']):
        os.makedirs(lda_params['model_dir'])
    dictionary.save(lda_params['model_dir'] + 'lda.dict')
    gensim.corpora.MmCorpus.serialize(lda_params['model_dir'] + 'lda.mm', corpus)
    lda_model.save(lda_params['model_dir'] + 'lda.model')
    with open(lda_params['model_dir'] + 'lda.docs', 'wb') as docs_file:
        pickle.dump(documents, docs_file, pickle.HIGHEST_PROTOCOL)
    with open(lda_params['model_dir'] + 'lda_params.config', 'w') as config_file:
        config_file.write(str(lda_params))

if __name__ == '__main__':
    t0 = time.time()
    documents = read_documents()
    t1 = time.time()
    build_model(documents)
    t2 = time.time()
    with open(lda_params['model_dir'] + 'lda_performance.info', 'w') as perf_file:
        perf_file.write('Total time elapsed:\t\t\t\t{}\nTime spent reading documents:\t\t\t{}\nTime spent building LDA:\t\t\t{}'.format(t2-t0, t1-t0, t2-t1))
