#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from lxml import etree
import regex
import nltk
import string
import csv
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import multiprocessing as mp

import tei_tools

data_dir = 'data/raw/'
ns = {'tei': '{http://www.tei-c.org/ns/1.0}', 'w3': '{http://www.w3.org/XML/1998/namespace}'}
sid = SentimentIntensityAnalyzer()

# Keywords adapted versionof Tams, S., & Grover, V. (2010). The Effect of an IS Article's Structure on Its Impact. CAIS, 27, 10.
introduction_keywords = ['introduction']
background_keywords = ['background',
                       'literature review',
                       'review of',
                       'critical review']
theory_frontend_keywords = ['conceptual development',
                            'hypothesis development',
                            'research hypotheses',
                            'research model',
                            'research questions',
                            'theory',
                            'theoretical background',
                            'theoretical development',
                            'theoretical model',
                            'theoretical']
method_keywords = ['data collection',
                   'methodology',
                   'methods',
                   'model testing',
                   'procedure',
                   'research methodology']
implications_keywords = ['contribution',
                         'discussion',
                         'future research',
                         'implications',
                         'implications for future research',
                         'implications for practice',
                         'limitations',
                         'practical implications',
                         'recommendations',
                         'theoretical implications']

# Extension of Tams and Grover (2010):
theory_frontend_keywords.extend(['theoretical foundation',
                                 'conceptual foundation',
                                 'conceptual basis',
                                 'model and hypotheses',
                                 'prior research',
                                 'related research',
                                 'theoretical framing',
                                 'theoretical framework',
                                 'framework',
                                 'hypotheses',
                                 'conceptualizing',
                                 'defining',
                                 'hypotheses development',
                                 'related literature',
                                 'model development'])
method_keywords.extend(['method',
                        'research design',
                        'research framework',
                        'research method',
                        'robustness',
                        'hypothesis testing',
                        'literature survey',
                        'scale validation',
                        'measur',
                        'control variable',
                        'coding'])
results_keywords = ['analysis',
                    'findings',
                    'results',
                    'robustness']
implications_keywords.extend(['conclusion',
                              'further research',
                              'concluding remarks',
                              'research agenda'])
appendix_keywords = ['appendi',
                     'electronic companion']

def parse_author(author_string):
    authors = author_string.split(' and ')
    last_names = []
    for author in authors:
        last_names.append(author[:author.index(',')])
    return last_names

def build_citation_regex(authors, year):
    if len(authors) == 1:
        return(authors[0] + "'?s?,? (\(?" + str(year) + '\)?)?')
    elif len(authors) == 2:
        return(authors[0] + ' (&|and|&amp;) ' + authors[1] + "'?s?,? (\(?" + str(year) + '\)?)?')
    else:
        return(authors[0] + ' et al.?,? (\(?' + str(year) + '\)?)?')

def get_position_in_sentence(sentence):
    return sentence.index('REFERENCE')/(len(sentence)-len('REFERENCE'))

def get_sentiment(document):
    return sid.polarity_scores(document)

def is_textual_citation(sentence):
    return regex.search('\([^(\))|^(\()]*?REFERENCE[^(\()|^(\))]*?\)', sentence, regex.DOTALL) is None

def is_separate(sentence):
    before = regex.search('CITATION ?REFERENCE', sentence, regex.DOTALL)
    after = regex.search('REFERENCE ?CITATION', sentence, regex.DOTALL)
    return before is None and after is None

def get_popularity(sentence, marker='CITATION'):
    return sentence.count(marker)

def get_density(sentence):
    return get_popularity(sentence, marker='REFERENCE') / (get_popularity(sentence, marker='CITATION') + get_popularity(sentence, marker='REFERENCE'))

def get_pos_structure(sentence):
    tokenized = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokenized)
    pos = []
    for pos_tag in pos_tags:
        if pos_tag[0] == 'REFERENCE':
            pos.append(pos_tag[0])
        else:
            pos.append(pos_tag[1])
    return ' '.join([tag for tag in pos if tag not in string.punctuation])

def find_pos_patterns(pos_sentence):
    pattern_0 = regex.compile('^.*REFERENCE VB[DPZN].*$').match(pos_sentence) is not None
    pattern_1 = regex.compile('^.*VB[DPZ] VB[GN].*$').match(pos_sentence) is not None
    pattern_2 = regex.compile('^.*VB[DGPZN]? (RB[RS]? )*VBN.*$').match(pos_sentence) is not None
    pattern_3 = regex.compile('^.*MD (RB[RS]? )*VB (RB[RS]? )*VBN.*$').match(pos_sentence) is not None
    pattern_4 = regex.compile('^(RB[RS]? )*PRP (RB[RS]? )*V.*$').match(pos_sentence) is not None
    pattern_5 = regex.compile('^.*VBG (NNP )*(CC )*(NNP ).*$').match(pos_sentence) is not None
    return [pattern_0, pattern_1, pattern_2, pattern_3, pattern_4, pattern_5]

def has_comp_sup(pos_sentence):
    return regex.compile('RB[RS]').match(pos_sentence) is not None

def has_1st_3rd_prp(citation_sentence):
    tokenized = nltk.word_tokenize(citation_sentence)
    pos_tags = nltk.pos_tag(tokenized)
    for pos_tag in pos_tags:
        if pos_tag[1] == 'PRP':
            if pos_tag[0] in ['I', 'i', 'We', 'we']:
                return True
    return False

def get_position_in_document(whole_document_text, predecessor, sentence, successor):
    predecessor_position = whole_document_text.find(extract_sentence_part_without_REF_or_CIT(predecessor))
    sentence_position = whole_document_text.find(extract_sentence_part_without_REF_or_CIT(sentence))
    successor_position = whole_document_text.find(extract_sentence_part_without_REF_or_CIT(successor))
    positions = [x for x in [predecessor_position, sentence_position, successor_position] if x > 1]

    if len(positions) > 0:
        return round(np.mean(positions)/len(whole_document_text), 3)
    else:
        return ''

def get_full_headings(root):
    full_headings = []
    for head in root.iter(ns['tei'] + 'head'):
        if head.getparent() is not None:
            if head.getparent().tag == ns['tei'] + 'figure':
                continue
        if head.text is not None:
            full_headings.append(str.title(head.text).lower())
    return full_headings

def get_heading(p):
    heading = 'NA'
    div = p.getparent()

    try:
        heading = div.find(ns['tei'] + 'head').text
        return heading
    except:
        pass
    # sometimes, there might be no heading in the same div tag -> check previous div.
    try:
        heading = div.xpath("preceding::div")[-1].find(ns['tei'] + 'head').text
        return heading
    except:
        pass
    return heading

def ref_in_tableDesc(el, heading_title):
    if 'figDesc' in el.getparent().tag and 'table' in heading_title.lower():
        return True
    if 'head' in el.getparent().tag and 'table' in el.getparent().text.lower():
        return True
    else:
        return False

def ref_in_figDesc(el, heading_title):
    if 'figDesc' in el.getparent().tag and 'figure' in heading_title.lower():
        return True
    if 'head' in el.getparent().tag and 'figure' in el.getparent().text.lower():
        return True
    else:
        return False

def ref_in_heading(el, heading_title):
    if 'head' in el.getparent().tag:
        return True
    else:
        return False

def match_headings(full_headings):
    matched_headings = ['-'] * len(full_headings)
    for i in range(0,len(full_headings)-1):
        if any(x in full_headings[i] for x in introduction_keywords):
            matched_headings[i] = 'introduction'
        if any(x in full_headings[i] for x in background_keywords):
            matched_headings[i] = 'background'
        if any(x in full_headings[i] for x in theory_frontend_keywords):
            matched_headings[i] = 'theory_frontend'
        if any(x in full_headings[i] for x in method_keywords):
            matched_headings[i] = 'method'
        if any(x in full_headings[i] for x in results_keywords):
            matched_headings[i] = 'results'
        if any(x in full_headings[i] for x in implications_keywords):
            matched_headings[i] = 'implications'
        if any(x in full_headings[i] for x in appendix_keywords):
            matched_headings[i] = 'appendix'

    # fill gap between same-category headings
    last_category = '-'
    for i in range(0,len(matched_headings)-1):
        if matched_headings[i] == '-':
            continue
        # now, we have cases in which matched_headings[i] != '-'
        # replace last_category if it differs from current heading
        if last_category != matched_headings[i]:
            last_category = matched_headings[i]
        # fill previous missing categories ('-') if previous category is the same
        else:
            n = i
            while True:
                matched_headings[n] = last_category
                n -= 1
                if matched_headings[n] != '-':
                    break

    # continue with same category if next category corresponds to IMRAD (intro background theory methods results discussion)
    last_category = '-'
    for i in range(0,len(matched_headings)-1):
        if matched_headings[i] == '-':
            continue
        # now, we have cases in which matched_headings[i] != '-'
        if last_category == 'introduction' and matched_headings[i] in ['background', 'theory_frontend']:
            n = i-1
            while n>=0:
                matched_headings[n] = last_category
                n -= 1
                if matched_headings[n] != '-':
                    break
        if last_category == 'background' and matched_headings[i] in ['theory_frontend']:
            n = i-1
            while n>=0:
                matched_headings[n] = last_category
                n -= 1
                if matched_headings[n] != '-':
                    break
        if last_category == 'theory_frontend' and matched_headings[i] in ['method']:
            n = i-1
            while n>=0:
                matched_headings[n] = last_category
                n -= 1
                if matched_headings[n] != '-':
                    break
        if last_category == 'method' and matched_headings[i] in ['results']:
            n = i-1
            while n>=0:
                matched_headings[n] = last_category
                n -= 1
                if matched_headings[n] != '-':
                    break
        if last_category == 'results' and matched_headings[i] in ['implications']:
            n = i-1
            while n>=0:
                matched_headings[n] = last_category
                n -= 1
                if matched_headings[n] != '-':
                    break
        last_category = matched_headings[i]

    # if last heading is an appendix: the following ones are also appendices
    n = len(matched_headings)
    while n >=1:
        n -= 1
        if matched_headings[n] == '-':
            continue
        if matched_headings[n] == 'appendix':
            if n != len(matched_headings)-1:
                while n < len(matched_headings):
                    matched_headings[n] = 'appendix'
                    n += 1
                break
        else:
            break

    return matched_headings

def get_heading_category(heading_title, position_in_document, full_headings, matched_headings):
    heading_catetory = 'NA'
    if heading_title is None:
        return heading_catetory
    for i in range(0,len(full_headings)):
        if heading_title.lower() == full_headings[i]:
            heading_catetory = matched_headings[i]
    if str(position_in_document).replace('.','').isdigit():
        if heading_title == 'NA' and position_in_document < 0.3:
            heading_catetory = 'introduction'
    return heading_catetory

def parse_numeric_citation(row, CURRENT_LR, root):
    df = pd.DataFrame(columns=columnnames)

    whole_document_text = str(etree.tostring(root.find('.//' + ns['tei'] + 'body'), pretty_print=True).decode('utf-8'))

    full_headings = get_full_headings(root)
    matched_headings = match_headings(full_headings)

    BIBLIOGRAPHY = pd.DataFrame(columns = ['reference_id', 'author', 'title', 'year', 'journal', 'similarity'])
    for reference in root.find('.//' + ns['tei'] + 'listBibl'):
        reference_id = tei_tools.get_reference_bibliography_id(reference)
        title_string = tei_tools.get_reference_title_string(reference)
        author_string = tei_tools.get_reference_author_string(reference)
        year_string = tei_tools.get_reference_year_string(reference)
        journal_string = tei_tools.get_reference_journal_string(reference)

        if title_string is None:
            continue

        ENTRY = pd.DataFrame.from_records([[reference_id, author_string, title_string, year_string, journal_string, 0]],
                                          columns = ['reference_id', 'author', 'title', 'year', 'journal', 'similarity'])

        ENTRY.loc[0, 'similarity'] = tei_tools.get_similarity(ENTRY, CURRENT_LR)
        BIBLIOGRAPHY = BIBLIOGRAPHY.append(ENTRY)

    BIBLIOGRAPHY = BIBLIOGRAPHY.reset_index(drop=True)

    LR_ENTRY = BIBLIOGRAPHY.loc[BIBLIOGRAPHY['similarity'].idxmax()]

    if LR_ENTRY['similarity'] > 0.85:
        ref_id = LR_ENTRY['reference_id']
        for ref in root.iter(ns['tei'] + 'ref'):
            if ref.get('target') == '#' + ref_id:
                p = ref.getparent()
                temp_p = etree.fromstring(etree.tostring(p))
                for elem in temp_p.iter(ns['tei'] + 'ref'):
                    if elem.get('target') != '#' + ref_id:
                        temp_p.text += 'CITATION'
                        if elem.tail:
                            temp_p.text += elem.tail
                        temp_p.remove(elem)
                    else:
                        temp_p.text += 'REFERENCE'
                        if elem.tail:
                            temp_p.text += elem.tail
                        temp_p.remove(elem)

                replacements =  {'c.f.':'cf', 'e.g.':'eg', 'pp.':'', 'etc.':'etc', 'cf.':'cf', '\n':'', '\r':''}
                for i, j in replacements.items():
                    temp_p.text = temp_p.text.replace(i, j)
                sentences = nltk.sent_tokenize(temp_p.text)

                for index, sentence in enumerate(sentences):
                    if 'REFERENCE' in sentence:
                        if index-1 < 0:
                            predecessor = ''
                        else:
                            predecessor = sentences[index-1]
                        if index+1 >= len(sentences):
                            successor = ''
                        else:
                            successor = sentences[index+1]
                        sentence = sentence.strip()
                        predecessor = predecessor.strip()
                        successor = successor.strip()

                        context = ' '.join([predecessor, sentence, successor])
                        sentence_sent = get_sentiment(sentence)
                        context_sent = get_sentiment(context)
                        pos_structure = get_pos_structure(sentence)
                        pos_patterns = find_pos_patterns(pos_structure)
                        position_in_document = get_position_in_document(whole_document_text, predecessor, sentence, successor)
                        heading_title = get_heading(p)

                        df.loc[len(df)] = [row['citation_key_lr'],
                               row['citation_key_cp'],
                               sentence,
                               predecessor,
                               successor,
                               False, # alphanumeric citations cannot be textual
                               is_separate(sentence),
                               get_popularity(sentence),
                               get_popularity(context),
                               get_density(sentence),
                               get_density(context),
                               get_position_in_sentence(sentence),
                               sentence_sent['neg'],
                               sentence_sent['neu'],
                               sentence_sent['pos'],
                               sentence_sent['compound'],
                               context_sent['neg'],
                               context_sent['neu'],
                               context_sent['pos'],
                               context_sent['compound'],
                               has_comp_sup(pos_structure),
                               has_1st_3rd_prp(sentence),
                               pos_structure,
                               pos_patterns[0],
                               pos_patterns[1],
                               pos_patterns[2],
                               pos_patterns[3],
                               pos_patterns[4],
                               pos_patterns[5],
                               position_in_document,
                               heading_title,
                               get_heading_category(heading_title, position_in_document, full_headings, matched_headings),
                               ref_in_figDesc(ref, heading_title),
                               ref_in_tableDesc(ref, heading_title),
                               ref_in_heading(ref, heading_title)]
    return(df)

def extract_sentence_part_without_REF_or_CIT(sentence):
    #always choose the shorter part since the longer includes the other type of marker
    left_reference_part = sentence[:sentence.find('REFERENCE')]
    left_citation_part = sentence[:sentence.find('CITATION')]
    if len(left_reference_part) > len(left_citation_part):
        left_part = left_citation_part
    else:
        left_part = left_reference_part
    right_reference_part = sentence[sentence.rfind('REFERENCE'):]
    right_citation_part = sentence[sentence.rfind('CITATION'):]
    if len(right_reference_part) > len(right_citation_part):
        right_part = right_citation_part
    else:
        right_part = right_reference_part
    #return the longer part since no markers included in left_part or right_part
    if len(left_part) > len(right_part):
        return left_part
    else:
        return right_part

def parse_standard_citation(row, CURRENT_LR, root):
    df = pd.DataFrame(columns=columnnames)
    citation_regex = build_citation_regex(parse_author(row['author_lr']), row['year_lr'])

    whole_document_text = str(etree.tostring(root.find('.//' + ns['tei'] + 'body'), pretty_print=True).decode('utf-8'))

    full_headings = get_full_headings(root)
    matched_headings = match_headings(full_headings)

    ref_id_lr = tei_tools.get_reference_id(root, CURRENT_LR)

    for ref in root.iter(ns['tei'] + 'ref'):
        if ref.text is not None:
            search_citation_regex = regex.search(citation_regex, ref.text, regex.DOTALL)

            search_grobid_id = False
            if ref_id_lr is not None and ref.get('target') is not None:
                search_grobid_id = ref.get('target').replace('#', '') == ref_id_lr

            if search_citation_regex or search_grobid_id:
                p = ref.getparent()

                if p.tag == ns['tei'] + 'div':
                    p.remove(ref)
                    p.find(ns['tei'] + 'p').insert(0, ref)
                    temp_p = etree.fromstring(etree.tostring(p.find(ns['tei'] + 'p').decode('utf-8')))
                else:
                    temp_p = etree.fromstring(etree.tostring(p))
                if temp_p.text is None:
                    continue
                for elem in temp_p.iter(ns['tei'] + 'ref'):
                    ref_search_citation_regex = regex.search(citation_regex, elem.text, regex.DOTALL)
                    ref_search_grobid_id = False
                    if ref_search_grobid_id is not None and elem.get('target') is not None:
                        ref_search_grobid_id = elem.get('target').replace('#', '') == ref_id_lr

                    if ref_search_citation_regex or ref_search_grobid_id:
                        temp_p.text += 'REFERENCE'
                        if elem.tail:
                            temp_p.text += elem.tail
                        temp_p.remove(elem)
                    else:
                        temp_p.text += 'CITATION'
                        if elem.tail:
                            temp_p.text += elem.tail
                        temp_p.remove(elem)

                replacements =  {'c.f.':'cf', 'e.g.':'eg', 'pp.':'', 'etc.':'etc', 'cf.':'cf', '\n':'', '\r':''}
                for i, j in replacements.items():
                    temp_p.text = temp_p.text.replace(i, j)
                sentences = nltk.sent_tokenize(temp_p.text)

                for index, sentence in enumerate(sentences):
                    if 'REFERENCE' in sentence:
                        if index-1 < 0:
                            predecessor = ''
                        else:
                            predecessor = sentences[index-1]
                        if index+1 >= len(sentences):
                            successor = ''
                        else:
                            successor = sentences[index+1]
                        sentence = sentence.strip()
                        predecessor = predecessor.strip()
                        successor = successor.strip()
                        context = ' '.join([predecessor, sentence, successor])
                        sentence_sent = get_sentiment(sentence)
                        context_sent = get_sentiment(context)
                        pos_structure = get_pos_structure(sentence)
                        pos_patterns = find_pos_patterns(pos_structure)
                        position_in_document = get_position_in_document(whole_document_text, predecessor, sentence, successor)
                        heading_title = get_heading(p)

                        df.loc[len(df)] = [row['citation_key_lr'],
                                           row['citation_key_cp'],
                                           sentence,
                                           predecessor,
                                           successor,
                                           is_textual_citation(sentence),
                                           is_separate(sentence),
                                           get_popularity(sentence),
                                           get_popularity(context),
                                           get_density(sentence),
                                           get_density(context),
                                           get_position_in_sentence(sentence),
                                           sentence_sent['neg'],
                                           sentence_sent['neu'],
                                           sentence_sent['pos'],
                                           sentence_sent['compound'],
                                           context_sent['neg'],
                                           context_sent['neu'],
                                           context_sent['pos'],
                                           context_sent['compound'],
                                           has_comp_sup(pos_structure),
                                           has_1st_3rd_prp(sentence),
                                           pos_structure,
                                           pos_patterns[0],
                                           pos_patterns[1],
                                           pos_patterns[2],
                                           pos_patterns[3],
                                           pos_patterns[4],
                                           pos_patterns[5],
                                           position_in_document,
                                           heading_title,
                                           get_heading_category(heading_title, position_in_document, full_headings, matched_headings),
                                           ref_in_figDesc(ref, heading_title),
                                           ref_in_tableDesc(ref, heading_title),
                                           ref_in_heading(ref, heading_title)]
    return(df)



def parse_citation(row):
    CURRENT_LR = ARTICLE[ARTICLE.citation_key == row['citation_key_lr']].head(1)
    CURRENT_LR = CURRENT_LR[['citation_key', 'author', 'title', 'year', 'journal']]
    CURRENT_LR.rename(index=str, columns={"citation_key": "reference_id"}, inplace=True)
    CURRENT_LR['similarity'] = 0

    # before parsing in-text citations: add ref-tags for LRs that have not been annotated by grobid
    file = open(data_dir + 'xml/' + row['citation_key_cp'] + '.tei.xml', "r")
    xml_string = file.read()
    root = etree.fromstring(xml_string)
    reference_id = tei_tools.get_reference_id(root, CURRENT_LR)
    author_list = parse_author(CURRENT_LR.iloc[0]['author'])
    if len(author_list) > 1:
        in_text_citation = build_citation_regex(parse_author(CURRENT_LR.iloc[0]['author']), CURRENT_LR.iloc[0]['year'])
        pattern = re.compile('(?!<ref[^>]*?>)(' + in_text_citation + ')(?![^<]*?</ref>)', re.IGNORECASE)
        main_part = xml_string.split('<listBibl>', 1)[0]
        reference_part = xml_string.split('<listBibl>', 1)[1]
        xml_string = pattern.sub('<ref target="#' + reference_id + '">\\1</ref>', main_part) + '<listBibl>' + reference_part

    # annotate cases like "D&M model
    if len(author_list) == 2:
        in_text_citation = author_list[0][0] + '&amp;' + author_list[1][0]
        pattern = re.compile('(?!<ref[^>]*?>)(' + in_text_citation + ')(?![^<]*?</ref>)', re.IGNORECASE)
        main_part = xml_string.split('<listBibl>', 1)[0]
        reference_part = xml_string.split('<listBibl>', 1)[1]
        xml_string = pattern.sub('<ref target="#' + reference_id + '">\\1</ref>', main_part) + '<listBibl>' + reference_part


    #    outfile = open("file.txt", 'w', encoding='utf-8')
    #    outfile.write(xml_string)
    #    outfile.close()

    root = etree.fromstring(str.encode(xml_string))

    if tei_tools.paper_alphanumeric_citation_style(root):
        result = parse_numeric_citation(row, CURRENT_LR, root)
    else:
        result = parse_standard_citation(row, CURRENT_LR, root)

    if result.empty:
        emptyvalues = [row['citation_key_lr'],
                       row['citation_key_cp'],
                       '', '', '', '', '', '', '', '', '', '',
                       '', '', '', '', '', '', '', '', '', '',
                       '', '', '', '', '', '', '', '', '', '',
                       '', '', '']
        df = pd.DataFrame(columns=columnnames)
        df.loc[0] = emptyvalues
        return(df)
    else:
        return(result)

def collect_result(result):
    global CITATION
    CITATION = pd.concat([CITATION, result])

if __name__ == "__main__":
    ARTICLE = pd.read_csv(data_dir + 'ARTICLE.csv')
    LR_CP = pd.read_csv(data_dir + 'LR_CP.csv')
    LR_CP = pd.merge(LR_CP, ARTICLE, left_on='citation_key_lr', right_on='citation_key')
    LR_CP = LR_CP[['citation_key_lr', 'citation_key_cp', 'title', 'author', 'year']]
    LR_CP.columns = ['citation_key_lr', 'citation_key_cp', 'title_lr', 'author_lr', 'year_lr']
    LR_CP = pd.merge(LR_CP, ARTICLE, left_on='citation_key_cp', right_on='citation_key')
    LR_CP = LR_CP[['citation_key_lr', 'citation_key_cp', 'title_lr', 'author_lr', 'year_lr', 'journal']]
    LR_CP.columns = ['citation_key_lr', 'citation_key_cp', 'title_lr', 'author_lr', 'year_lr', 'journal_cp']
    columnnames = ['citation_key_lr',
                   'citation_key_cp',
                   'citation_sentence',
                   'predecessor',
                   'successor',
                   'textual',
                   'separate',
                   'sentence_popularity',
                   'context_popularity',
                   'sentence_density',
                   'context_density',
                   'position_in_sentence',
                   'sentence_neg',
                   'sentence_neu',
                   'sentence_pos',
                   'sentence_compound',
                   'context_neg',
                   'context_neu',
                   'context_pos',
                   'context_compound',
                   'comp_sup',
                   'prp',
                   'pos_pattern',
                   'pos_0',
                   'pos_1',
                   'pos_2',
                   'pos_3',
                   'pos_4',
                   'pos_5',
                   'position_in_document',
                   'heading_title',
                   'heading_category',
                   'ref_in_figure_description',
                   'ref_in_table_description',
                   'ref_in_heading']
    CITATION = pd.DataFrame(columns=columnnames)

    pool = mp.Pool(mp.cpu_count()-2)
    for i, row in LR_CP.iterrows():
        pool.apply_async(parse_citation, args=(row, ), callback=collect_result)
    pool.close()
    pool.join()
    
    CITATION = CITATION.drop_duplicates()
    CITATION = CITATION.sort_values(['citation_key_lr', 'citation_key_cp'])
    CITATION.to_csv('data/interim/CITATION.csv', index=False, quoting=csv.QUOTE_ALL)
