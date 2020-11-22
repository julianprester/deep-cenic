#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lxml import etree
import re
import pandas as pd
from fuzzywuzzy import fuzz

ns = {'tei': '{http://www.tei-c.org/ns/1.0}', 'w3': '{http://www.w3.org/XML/1998/namespace}'}

def paper_alphanumeric_citation_style(root):
    alphanumeric_references = []
    for reference in root.iter(ns['tei'] + 'ref'):
            if reference.get('type') == 'bibr' and reference.text is not None:
                # years in brackets are an indicator of non-alphanumeric styles (delete them)
                ref_text = re.sub('\([1-3][0-9]{3}\)', '', reference.text)
                if len(ref_text) > 1:
                    nr_numbers = len(re.sub('[^0-9]', '', ref_text))
                    nr_letters = len(re.sub('[^a-zA-Z]', '', ref_text))
                    # heutistic: number of references in which there are fewer letters than numbers    
                    if nr_numbers > nr_letters:
                        alphanumeric_references.append(True)
                    else:
                        alphanumeric_references.append(False)
    if alphanumeric_references.count(True) > alphanumeric_references.count(False):
        return True
    else:
        return False

def extract_abstract(root):
    abstract = root.find('.//' + ns['tei'] + 'abstract')

    if abstract is None:
        return None
    else:
        if abstract.find('.//' + ns['tei'] + 'div') is None:
            return None
        else:
            if abstract.find('.//' + ns['tei'] + 'p') is None:
                return None
            else:
                if abstract.find('.//' + ns['tei'] + 'p').text is None:
                    return None
                else:
                    if not abstract.find('.//' + ns['tei'] + 'p').text.strip().replace('\n','').replace('\r','') == '':
                        text = ''.join(abstract.itertext()).strip().replace('\n','').replace('\r','')
                        return text
    return None

def get_reference_id(root, REFERENCE):

    BIBLIOGRAPHY = pd.DataFrame(columns = ['reference_id', 'author', 'title', 'year', 'journal', 'similarity'])

    bibliographies = root.iter(ns['tei'] + 'listBibl')
    for bibliography in bibliographies:
        for reference in bibliography:

            reference_id = get_reference_bibliography_id(reference)
            title_string = get_reference_title_string(reference)
            author_string = get_reference_author_string(reference)
            year_string = get_reference_year_string(reference)
            journal_string = get_reference_journal_string(reference)

            if(title_string is None and journal_string and len(journal_string) > 0):
                title_string = journal_string

            if title_string is not None:
                ENTRY = pd.DataFrame.from_records([[reference_id, author_string, title_string, year_string, journal_string, 0]], columns = ['reference_id', 'author', 'title', 'year', 'journal', 'similarity'])

                ENTRY.loc[0, 'similarity'] = get_similarity(ENTRY, REFERENCE)
                BIBLIOGRAPHY = BIBLIOGRAPHY.append(ENTRY)

    BIBLIOGRAPHY = BIBLIOGRAPHY.reset_index(drop=True)
    if BIBLIOGRAPHY.shape[0] == 0:
        return 'no_bibliography'
    if BIBLIOGRAPHY['similarity'].max() < 0.8:
        return 'not_found'

    reference_id = BIBLIOGRAPHY.loc[BIBLIOGRAPHY['similarity'].idxmax(), 'reference_id']
    return reference_id


def get_similarity(df_a, df_b):
    # df_a:= extracted from PDF
    # df_b:= literature review
    authors_a = re.sub(r'[^A-Za-z0-9, ]+', '', str(df_a['authors']).lower())
    authors_b = re.sub(r'[^A-Za-z0-9, ]+', '', str(df_b['authors']).lower())
    author_similarity = fuzz.ratio(authors_a, authors_b)/100

    #partial ratio (catching 2010-10 or 2001-2002)
    year_similarity = fuzz.partial_ratio(str(df_a['year']), str(df_b['year']))/100

    journal_a = re.sub(r'[^A-Za-z0-9 ]+', '', str(df_a['journal']).lower())
    journal_b = re.sub(r'[^A-Za-z0-9 ]+', '', str(df_b['journal']).lower())
    
    # replacing abbreviations before matching and matching lower cases (catching different citation styles)
    # for i, row in JOURNAL_ABBREVIATIONS.iterrows():
    #     if journal_a == row['abbreviation']:
    #         journal_a = row['journal']
    #     if journal_b == row['abbreviation']:
    #         journal_b = row['journal']
    journal_similarity = fuzz.ratio(journal_a, journal_b)/100

    title_a = df_a['title'].str.lower().replace(regex={'information technology':'it', 'information systems':'is', 'resource-based view':'rbv', r'^review':'', r'[^A-Za-z0-9, ]+':''})
    title_b = df_b['title'].str.lower().replace(regex={'information technology':'it', 'information systems':'is', 'resource-based view':'rbv', r'^review':'', r'[^A-Za-z0-9, ]+':''})
    
    # titles are sometimes (errorneously) in the journal-fields...
    title_similarity = max(fuzz.ratio(title_a, title_b)/100, fuzz.ratio(journal_a, title_b)/100)
    if fuzz.ratio(journal_a, title_b)/100 > 0.9:
        journal_similarity = 1

    weights = [0.15, 0.75, 0.05, 0.05]
    similarities = [author_similarity, title_similarity, year_similarity, journal_similarity]
    weighted_average = sum(similarities[g] * weights[g] for g in range(len(similarities)))

    return weighted_average


# paper metadata -----------------------------------------------

def get_paper_title(root):
    title_text = 'NA'
    try:
        file_description = root.find('.//' + ns['tei'] + 'fileDesc')
        title_text = file_description.find('.//' + ns['tei'] + 'title').text
    except:
        pass
    return title_text

# (individual) bibliography-reference elements  --------------------------------------------------------------------------------------


def get_reference_author_string(reference):
    author_list = []
    if reference.find(ns['tei'] + 'analytic') is not None:
        author_node = reference.find(ns['tei'] + 'analytic')
    elif reference.find(ns['tei'] + 'monogr') is not None:
        author_node = reference.find(ns['tei'] + 'monogr')

    for author in author_node.iterfind(ns['tei'] + 'author'):
        authorname = ''
        try:
            surname = author.find(ns['tei'] + 'persName').find(ns['tei'] + 'surname').text
        except:
            surname = ''
        pass
        try:
            forename = author.find(ns['tei'] + 'persName').find(ns['tei'] + 'forename').text
        except:
            forename = ''
        pass

        #check surname and prename len. and swap
        if(len(surname) < len(forename)):
            authorname = forename + ', ' + surname
        else:
            authorname = surname + ', ' + forename
        author_list.append(authorname)

    #fill author field with editor or organization if null
    if len(author_list) == 0:
        if reference.find(ns['tei'] + 'editor') is not None:
            author_list.append(reference.find(ns['tei'] + 'editor').text)
        elif reference.find(ns['tei'] + 'orgName') is not None:
            author_list.append(reference.find(ns['tei'] + 'orgName').text)

    author_string = ''
    for author in author_list:
        author_string = ';'.join(author_list)
    author_string = author_string.replace('\n', ' ').replace('\r', '')

    if author_string is None:
        author_string = 'NA'

    return author_string

def get_reference_title_string(reference):
    title_string = ''
    if reference.find(ns['tei'] + 'analytic') is not None:
        title = reference.find(ns['tei'] + 'analytic').find(ns['tei'] + 'title')
    elif reference.find(ns['tei'] + 'monogr') is not None:
        title = reference.find(ns['tei'] + 'monogr').find(ns['tei'] + 'title')
    if title is None:
        title_string = 'NA'
    else:
        title_string = title.text
    return title_string

def get_reference_year_string(reference):
    year_string = ''
    if reference.find(ns['tei'] + 'monogr') is not None:
        year = reference.find(ns['tei'] + 'monogr').find(ns['tei'] + 'imprint').find(ns['tei'] + 'date')
    elif reference.find(ns['tei'] + 'analytic') is not None:
        year = reference.find(ns['tei'] + 'analytic').find(ns['tei'] + 'imprint').find(ns['tei'] + 'date')

    if not year is None:
        for name, value in sorted(year.items()):
            if name == 'when':
                year_string = value
            else:
                year_string = 'NA'
    else:
        year_string = 'NA'
    return year_string

def get_reference_journal_string(reference):
    journal_title = ''
    if reference.find(ns['tei'] + 'monogr') is not None:
        journal_title = reference.find(ns['tei'] + 'monogr').find(ns['tei'] + 'title').text
    if journal_title is None:
        journal_title = ''
    return journal_title

def get_reference_bibliography_id(reference):
    return reference.attrib[ns['w3'] + 'id']
