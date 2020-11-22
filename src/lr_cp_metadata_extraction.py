#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from lxml import etree
from gensim import corpora, models, matutils
import preprocessing
import re

import tei_tools

ns = {'tei': '{http://www.tei-c.org/ns/1.0}', 'w3': '{http://www.w3.org/XML/1998/namespace}'}

title_dict = corpora.Dictionary.load('models/lsa/title.dict')
title_lsi = models.LsiModel.load('models/lsa/title.model')

abstract_dict = corpora.Dictionary.load('models/lsa/abstract.dict')
abstract_lsi = models.LsiModel.load('models/lsa/abstract.model')

def parse_author(author):
    result = []
    authors = author.split(' and ')
    for author in authors:
        result.append(author[:author.index(',')])
    return result

def is_self_citation(row):
    lr_author = parse_author(row['author_lr'])
    cp_author = parse_author(row['author_cp'])
    for author in cp_author:
        if author in lr_author:
            return True
    return False

def get_title_similarity(row):
    if pd.notnull(row['title_lr']) and pd.notnull(row['title_cp']):
        lr_doc = preprocess_doc(row['title_lr'])
        cp_doc = preprocess_doc(row['title_cp'])
        lr_bow = title_dict.doc2bow(lr_doc)
        cp_bow = title_dict.doc2bow(cp_doc)
        lr_lsi = title_lsi[lr_bow]
        cp_lsi = title_lsi[cp_bow]
        return matutils.cossim(lr_lsi, cp_lsi)
    else:
        return 0

def get_abstract_similarity(row):
    if pd.notnull(row['abstract_lr']) and pd.notnull(row['abstract_cp']):
        lr_doc = preprocess_doc(row['abstract_lr'])
        cp_doc = preprocess_doc(row['abstract_cp'])
        lr_bow = abstract_dict.doc2bow(lr_doc)
        cp_bow = abstract_dict.doc2bow(cp_doc)
        lr_lsi = abstract_lsi[lr_bow]
        cp_lsi = abstract_lsi[cp_bow]
        return matutils.cossim(lr_lsi, cp_lsi)
    else:
        return 0

def build_citation_regex(authors):
    if len(authors) == 1:
        return(authors[0])
    elif len(authors) == 2:
        return(authors[0] + ' (&|and) ' + authors[1])
    else:
        return(authors[0] + ' et al.')

def check_ref_in_title(root, authors):
    try:
        title_text = tei_tools.get_paper_title(root)
        
        lr_author_regex = re.compile(build_citation_regex(authors), re.IGNORECASE)
        if re.search(lr_author_regex, title_text) is not None:
            return True
        else:
            return False
    except:
        pass
        return False

def preprocess_doc(doc):
    doc = preprocessing.tokenize(doc)
    doc = preprocessing.remove_punctuation(doc)
    doc = preprocessing.remove_numbers(doc)
    doc = preprocessing.lower(doc)
    doc = preprocessing.remove_common_stopwords(doc)
    doc = preprocessing.clean_doc(doc)
    return doc

def extract_lr_cp_data():
    ARTICLE = pd.read_csv('data/raw/ARTICLE.csv')
    ARTICLE.drop(columns=['title', 'year', 'journal', 'volume', 'issue', 'pages'], inplace=True)
    LR_CP = pd.read_csv('data/raw/LR_CP.csv')
    LR = pd.read_csv('data/interim/LR.csv')
    LR = LR[['citation_key_lr', 'title', 'abstract']]
    LR.rename(columns = {'title' : 'title_lr', 'abstract': 'abstract_lr'}, inplace=True)
    CP = pd.read_csv('data/interim/CP.csv')
    CP = CP[['citation_key_cp', 'title', 'abstract']]
    CP.rename(columns = {'title': 'title_cp', 'abstract': 'abstract_cp'}, inplace=True)
    LR_CP = pd.merge(LR_CP, ARTICLE, left_on='citation_key_lr', right_on='citation_key')
    LR_CP = LR_CP[['citation_key_lr', 'citation_key_cp', 'author', 'NOT', 'SYN_TB', 'CRI_ADDR', 'RG_SYN', 'RG_CLOSE', 'RA_CLOSE', 'TB_TB', 'TB_TT', 'TB_RG', 'TT_TT', 'TT_RG']]
    LR_CP.rename(columns = {'author': 'author_lr'}, inplace=True)
    LR_CP = pd.merge(LR_CP, ARTICLE, left_on='citation_key_cp', right_on='citation_key')
    LR_CP.rename(columns = {'author': 'author_cp'}, inplace=True)
    LR_CP['self_citation'] = False
    LR_CP = pd.merge(LR_CP, LR, on='citation_key_lr')
    LR_CP = pd.merge(LR_CP, CP, on='citation_key_cp')
    
    for index, row in LR_CP.iterrows():
        root = etree.parse('data/raw/xml/' + row['citation_key_cp'] + '.tei.xml').getroot()
        LR_CP.loc[index, 'self_citation'] = is_self_citation(row)
        LR_CP.loc[index, 'title_similarity'] = get_title_similarity(row)
        LR_CP.loc[index, 'abstract_similarity'] = get_abstract_similarity(row)
        LR_CP.loc[index, 'ref_in_title'] = check_ref_in_title(root, parse_author(row['author_lr']))

    #rearrange order
    LR_CP = LR_CP[['citation_key_lr', 'citation_key_cp' , 'self_citation', 'title_similarity', 'abstract_similarity', 'ref_in_title', 'NOT', 'SYN_TB', 'CRI_ADDR', 'RG_SYN', 'RG_CLOSE', 'RA_CLOSE', 'TB_TB', 'TB_TT', 'TB_RG', 'TT_TT', 'TT_RG']]

    LR_CP.to_csv('data/interim/LR_CP.csv', index=False)

if __name__ == "__main__":
    extract_lr_cp_data()
