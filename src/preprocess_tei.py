#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lxml import etree
from wordsegment import load, segment
from nltk.corpus import words
import pandas as pd
import re
from fuzzywuzzy import fuzz

data_dir = 'data/raw/'

ns = {'tei': '{http://www.tei-c.org/ns/1.0}', 'w3': '{http://www.w3.org/XML/1998/namespace}'}
nsmap = {'tei': 'http://www.tei-c.org/ns/1.0', 'w3': 'http://www.w3.org/XML/1998/namespace'}

load()

LR_CP = pd.read_csv(data_dir + 'LR_CP.csv')

for i, row in LR_CP.iterrows():
    with open(data_dir + 'xml/' + row['citation_key_cp'] + '.tei.xml') as xml_file:
        root = etree.parse(xml_file).getroot()
    
    # do not save date of Grobid conversion (for cleaner git versioning)
    try:
        app_node = root.find('.//' + ns['tei'] + 'application')
        app_node.attrib.pop('when')
    except:
        pass
    
    for head in root.iter(ns['tei'] + 'head'):
        if not head.text is None:
            if len(head.text) > 10  and head.text.count(' ')/len(head.text) > 0.3:
                head.text = ' '.join(segment(head.text))
            # fix cases in which a capitalized first letter (and an optional blank space) are annotated as a heading
            if len(head.text) < 5 and not head.text in words.words():
                temp = head.text
                parent_element = head.getparent()
                if head.getnext() is not None:
                    if head.getnext().text is not None:
                        head.getnext().text = temp + head.getnext().text
                        parent_element.remove(head)
    
    
    # If abstracts are too long (errorneously include parts of the first section): move content to the first section (heuristic: first p-tag that includes a reference)
    p_list_to_move = []
    div_list_to_move = []
    for abstract in root.iter(ns['tei'] + 'abstract'):
        if len(etree.tostring(abstract).decode()) > 3000:
            move_following_ps = False # as soon as this is true, move all of the following
            #append to a list, then add the list to the first p of the body (to keep the same order!)
            for div in abstract.iter(ns['tei'] + 'div'):
                if not move_following_ps:
                    for p in div.iter(ns['tei'] + 'p'):
                        nr_references_in_p = sum(1 for refer in p.iter(ns['tei'] + 'ref') if refer.attrib['type'] == 'bibr')
                        if move_following_ps:
                            p_list_to_move.append(p)
                            div.remove(p)
                            continue
                        if nr_references_in_p  > 0:
                            p_list_to_move.append(p)
                            div.remove(p)
                            move_following_ps = True
                else:
                    div_list_to_move.append(div)
                    abstract.remove(div)

        # drop p from p_list_to_move if a p with (almost) identical contents is already contained in the body
        for p in p_list_to_move:
            for body in root.iter(ns['tei'] + 'body'):
                for p_existing in body.iter(ns['tei'] + 'p'):
                    if fuzz.ratio(p.text, p_existing.text) > 95:
                        p_list_to_move.remove(p)

        # drop div from div_list_to_move if a div with (almost) identical contents is already contained in the body
        for div in div_list_to_move:            
            for p in div.iter(ns['tei'] + 'p'):
                for body in root.iter(ns['tei'] + 'body'):
                    for p_existing in body.iter(ns['tei'] + 'p'):
                        if fuzz.ratio(p.text, p_existing.text) > 95:
                            if div in div_list_to_move:
                                div_list_to_move.remove(div)
            break # i.e., consider only the first paragraph

        for i in range(len(div_list_to_move), 0):
            body.insert(0,div_list_to_move[i])
        
        new_first_div = etree.Element(ns['tei'] + "div", nsmap = nsmap)
        for p in p_list_to_move:
            new_first_div.insert(0,p)
        for body in root.iter(ns['tei'] + 'body'):
            for div in body.iter(ns['tei'] + 'div'):
                for p in div.iter(ns['tei'] + 'p'):
                    if p.text is not None:
                        for add_text in p_list_to_move:
                            p.insert(0, add_text)
                break
            break
            
    # Remove author information from abstract
    abstract_div = root.find('.//' + ns['tei'] + 'abstract').find('.//' + ns['tei'] + 'div')
    if abstract_div is not None:
        p_nr = 0
        p_nr_abstract = 0
        for p in abstract_div.iter(ns['tei'] + 'p'):
            p_nr += 1
            if p.text is not None:
                abstract_pattern = re.compile('a\s?b\s?s\s?t\s?r\s?a\s?c\s?t:?', re.IGNORECASE)    
                pos_begining_abstract = re.search(abstract_pattern, p.text)
                if pos_begining_abstract is not None:
                    p.text = p.text[pos_begining_abstract.start():]
                    p_nr_abstract = p_nr
                    p.text = abstract_pattern.sub('', p.text)
        p_nr = 0
        if p_nr_abstract != 0:
            for p in abstract_div.iter(ns['tei'] + 'p'):
                p_nr += 1
                if p_nr == p_nr_abstract:
                    break
                parent_element = p.getparent()
                parent_element.remove(p)
        first_body_p = root.find('.//' + ns['tei'] + 'body').find('.//' + ns['tei'] + 'div').find('.//' + ns['tei'] + 'p')
        # if key words and phrases is in the first p of the body, everything before that belongs to the abstract
        if first_body_p is not None:
            if first_body_p.text is not None:
                pos_begining_abstract = re.search('a\s?b\s?s\s?t\s?r\s?a\s?c\s?t:? ', first_body_p.text.lower())
                if pos_begining_abstract is not None:
                    if abstract_div.find('.//' + ns['tei'] + 'p') is not None:
                        if abstract_div.find('.//' + ns['tei'] + 'p').text is not None:
                            abstract_div.find('.//' + ns['tei'] + 'p').text = ''
                if 'key words and phrases' in first_body_p.text.lower():
                    pos_abstract_end = first_body_p.text.lower().find('key word')
                    end_of_abstract = first_body_p.text[:pos_abstract_end]
                    p_nr_abstract = sum(1 for  _ in abstract_div.iter(ns['tei'] + 'p'))
                    p_nr = 0
                    for p in abstract_div.iter(ns['tei'] + 'p'):
                        p_nr += 1
                        if p_nr == p_nr_abstract:
                            p.text = p.text + ' ' + end_of_abstract
                    first_body_p.text = first_body_p.text.replace(end_of_abstract, '')
        
    # No headings in abstract
    for abstract in root.iter(ns['tei'] + 'abstract'):
        for div in abstract.iter(ns['tei'] + 'div'):
            for head in abstract.iter(ns['tei'] + 'head'):
                temp = head.text
                if 'abstract' in head.text.lower():
                    continue
                parent_element = head.getparent()
                if head.getnext() is not None:
                    if head.getnext().text is not None:
                        head.getnext().text = temp + head.getnext().text
                        parent_element.remove(head)         

    tree = etree.ElementTree(root)
    tree.write(data_dir + 'xml/' + row['citation_key_cp'] + '.tei.xml', pretty_print=True, encoding="utf-8")
    