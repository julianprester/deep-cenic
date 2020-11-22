#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from lxml import etree
import re

import tei_tools

data_dir = 'data/raw/'
ns = {'tei': '{http://www.tei-c.org/ns/1.0}', 'w3': '{http://www.w3.org/XML/1998/namespace}'}

def extract_total_references(root):
    references = root.findall('.//' + ns['tei'] + 'div[@type="references"]/' + ns['tei'] + 'listBibl/' + ns['tei'] + 'biblStruct')
    return len(references)

def extract_total_citations(root):
    citations = root.findall('.//' + ns['tei'] + 'ref[@type="bibr"]')
    return len(citations)

def get_abstract_replacement(root):
    replacement = ''
    try:
        div = root.find('.//' + ns['tei'] + 'body').find('.//' + ns['tei'] + 'div')
        replacement = str(etree.tostring(div, pretty_print=True).decode('utf-8'))
        clean = re.compile('<.*?>')
        replacement = re.sub(clean, '', replacement)
    except:
        pass
    replacement = replacement.replace('\n','').replace('\r','').lstrip().rstrip()
    return replacement

def extract_cp_data():
    ARTICLE = pd.read_csv(data_dir + 'ARTICLE.csv')
    LR_CP = pd.read_csv(data_dir + 'LR_CP.csv')
    LR_CP = LR_CP[['citation_key_cp']].drop_duplicates()
    LR_CP = pd.merge(LR_CP, ARTICLE, left_on='citation_key_cp', right_on='citation_key')
    LR_CP['total_references'] = 0
    LR_CP['total_citations'] = 0
    LR_CP['abstract'] = ''
    for index, row in LR_CP.iterrows():
        root = etree.parse(data_dir + 'xml/' + row['citation_key_cp'] + '.tei.xml').getroot()
        
        LR_CP.loc[index, 'total_references'] = extract_total_references(root)
        LR_CP.loc[index, 'total_citations'] = extract_total_citations(root)
        LR_CP.loc[index, 'abstract'] = (tei_tools.extract_abstract(root) 
                                        if (tei_tools.extract_abstract(root) is not None) 
                                        else get_abstract_replacement(root))
        
    LR_CP = LR_CP[['citation_key_cp', 'title', 'total_references', 'total_citations', 'abstract']]
    LR_CP.to_csv('data/interim/CP.csv', index=False)

if __name__ == "__main__":
    extract_cp_data()
