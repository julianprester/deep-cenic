#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import csv
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from config import lda_params
import preprocessing

def prepare_dataframe(df, context=True):
    df.dropna(subset=['citation_sentence'], inplace=True)
    df.fillna('', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if context:
        df['predecessor'] = df['predecessor'].astype(str)
        df['successor'] = df['successor'].astype(str)
        df['context'] = df['predecessor'] + ' ' + df['citation_sentence'] + ' ' + df['successor']
    else:
        df['context'] = df['citation_sentence']
    df = df.groupby(['citation_key_lr', 'citation_key_cp'])['context'].apply(lambda x: ' '.join(x)).reset_index()
    return df

def preprocess_doc(row, context=True):
    citation_sentence = str(row['context'])
    if lda_params['markers']:
        citation_sentence = preprocessing.remove_markers(citation_sentence)
    if lda_params['tokenize']:
        citation_sentence = preprocessing.tokenize(citation_sentence)
    if lda_params['pos_tags'] != ():
        tags = preprocessing.lower(preprocessing.filter_pos_tags(citation_sentence, tags=lda_params['pos_tags']))
    if lda_params['punctuation']:
        citation_sentence = preprocessing.remove_punctuation(citation_sentence)
    if lda_params['numbers']:
        citation_sentence = preprocessing.remove_numbers(citation_sentence)
    citation_sentence = preprocessing.lower(citation_sentence)
    if lda_params['bigrams']:
        bigrams = preprocessing.get_bigrams(citation_sentence)
    if lda_params['trigrams']:
        trigrams = preprocessing.get_trigrams(citation_sentence)
    if lda_params['common_stopwords']:
        citation_sentence = preprocessing.remove_common_stopwords(citation_sentence)
    if lda_params['custom_stopwords']:
        citation_sentence = preprocessing.remove_custom_stopwords(citation_sentence)
    if lda_params['pos_tags'] != ():
        citation_sentence = preprocessing.filter_pos(citation_sentence, tags)
    citation_sentence = preprocessing.clean_doc(citation_sentence)
    if lda_params['bigrams']:
        bigrams = preprocessing.filter_n_grams(bigrams, citation_sentence)
    if lda_params['trigrams']:
        trigrams = preprocessing.filter_n_grams(trigrams, citation_sentence)
    if lda_params['bigrams'] and not lda_params['trigrams']:
        citation_sentence = citation_sentence + bigrams
    if lda_params['trigrams'] and not lda_params['bigrams']:
        citation_sentence = citation_sentence + trigrams
    if lda_params['bigrams'] and lda_params['trigrams']:
        citation_sentence = citation_sentence + bigrams + trigrams
    if lda_params['lemmatize']:
        citation_sentence = preprocessing.lemmatize(citation_sentence)
    citation_sentence = preprocessing.clean_doc(citation_sentence)
    return citation_sentence

def summarize_citation_df(df):
    df = df[df['citation_sentence'].notnull()]
    df_keys = df.loc[:,['citation_key_lr', 'citation_key_cp']]
    df_keys['focal_citations'] = 0
    df_keys = df_keys.groupby(['citation_key_lr', 'citation_key_cp']).agg({'focal_citations': 'count'}).reset_index()

    df_bool = df.loc[:,['citation_key_lr', 
                        'citation_key_cp', 
                        'textual',
                        'separate', 
                        'comp_sup',
                        'prp',
                        'pos_0', 
                        'pos_1',
                        'pos_2', 
                        'pos_3', 
                        'pos_4', 
                        'pos_5', 
                        'ref_in_figure_description', 
                        'ref_in_table_description', 
                        'ref_in_heading']]
    df_bool = df_bool.groupby(['citation_key_lr', 'citation_key_cp']).aggregate(np.sum)
    df_bool = df_bool.reset_index()
    
    df_else = df[['citation_key_lr', 'citation_key_cp', 'sentence_popularity','context_popularity', 'sentence_density', 'context_density','position_in_sentence', 'sentence_neg', 'sentence_neu', 'sentence_pos','sentence_compound', 'context_neg', 'context_neu', 'context_pos','context_compound']]
    df_else = df_else.groupby(['citation_key_lr', 'citation_key_cp']).agg([np.min, np.max, np.mean])
    df_else = df_else.reset_index()
    df_else.columns = [' '.join(col).strip() for col in df_else.columns.values]

    df_mention_position = df[['citation_key_lr', 'citation_key_cp', 'position_in_document']].drop_duplicates().reset_index()
    df_mention_position['mention_positions_10'] = 0
    df_mention_position['mention_positions_20'] = 0
    df_mention_position['mention_positions_30'] = 0
    df_mention_position['mention_positions_40'] = 0
    df_mention_position['mention_positions_50'] = 0
    df_mention_position['mention_positions_60'] = 0
    df_mention_position['mention_positions_70'] = 0
    df_mention_position['mention_positions_80'] = 0
    df_mention_position['mention_positions_90'] = 0
    df_mention_position['mention_positions_100'] = 0

    for i,row in df_mention_position.iterrows():
        # test for NaN
        if row['position_in_document'] == '':
            continue
        mention_position = math.ceil(float(row['position_in_document'])*100) # mention position in the first x % of the paper
        if 0 < mention_position <= 10:
            df_mention_position.at[i, 'mention_positions_10'] +=1
        if 10 < mention_position <= 20:
            df_mention_position.at[i, 'mention_positions_20'] +=1
        if 20 < mention_position <= 30:
            df_mention_position.at[i, 'mention_positions_30'] +=1
        if 30 < mention_position <= 40:
            df_mention_position.at[i, 'mention_positions_40'] +=1
        if 40 < mention_position <= 50:
            df_mention_position.at[i, 'mention_positions_50'] +=1
        if 50 < mention_position <= 60:
            df_mention_position.at[i, 'mention_positions_60'] +=1
        if 60 < mention_position <= 70:
            df_mention_position.at[i, 'mention_positions_70'] +=1
        if 70 < mention_position <= 80:
            df_mention_position.at[i, 'mention_positions_80'] +=1
        if 80 < mention_position <= 90:
            df_mention_position.at[i, 'mention_positions_90'] +=1
        if 90 < mention_position <= 100:
            df_mention_position.at[i, 'mention_positions_100'] +=1

    df_mention_position = df_mention_position.groupby(['citation_key_lr', 'citation_key_cp']).agg([np.sum])
    df_mention_position = df_mention_position.reset_index()
    df_mention_position.columns = [' '.join(col).strip() for col in df_mention_position.columns.values]
    df_mention_position.drop(columns = ['index sum'], inplace=True)
    df_mention_position.rename(columns={'mention_positions_10 sum': 'mention_positions_10',
                                          'mention_positions_20 sum': 'mention_positions_20',
                                          'mention_positions_30 sum': 'mention_positions_30',
                                          'mention_positions_40 sum': 'mention_positions_40',
                                          'mention_positions_50 sum': 'mention_positions_50',
                                          'mention_positions_60 sum': 'mention_positions_60',
                                          'mention_positions_70 sum': 'mention_positions_70', 
                                          'mention_positions_80 sum': 'mention_positions_80', 
                                          'mention_positions_90 sum': 'mention_positions_90',
                                          'mention_positions_100 sum': 'mention_positions_100'}, inplace=True)


    df_heading_category = df.loc[:,['citation_key_lr', 'citation_key_cp', 'heading_category']].drop_duplicates().reset_index()
    df_heading_category['heading_category_NA'] = 0
    df_heading_category['heading_category_intro'] = 0
    df_heading_category['heading_category_background'] = 0
    df_heading_category['heading_category_theory'] = 0
    df_heading_category['heading_category_methods'] = 0
    df_heading_category['heading_category_results'] = 0
    df_heading_category['heading_category_implications'] = 0
    df_heading_category['heading_category_appendix'] = 0
    
    
    for i,row in df_heading_category.iterrows():
        # test for NaN
        if row['heading_category'] == '-':
            df_heading_category.at[i, 'heading_category_NA'] +=1
        if row['heading_category'] == 'introduction':
            df_heading_category.at[i, 'heading_category_intro'] +=1
        if row['heading_category'] == 'background':
            df_heading_category.at[i, 'heading_category_background'] +=1
        if row['heading_category'] == 'theory_frontend':
            df_heading_category.at[i, 'heading_category_theory'] +=1
        if row['heading_category'] == 'method':
            df_heading_category.at[i, 'heading_category_methods'] +=1
        if row['heading_category'] == 'results':
            df_heading_category.at[i, 'heading_category_results'] +=1
        if row['heading_category'] == 'implications':
            df_heading_category.at[i, 'heading_category_implications'] +=1
        if row['heading_category'] == 'appendix':
            df_heading_category.at[i, 'heading_category_appendix'] +=1
        
    df_heading_category = df_heading_category.groupby(['citation_key_lr', 'citation_key_cp']).agg([np.sum])
    df_heading_category = df_heading_category.reset_index()
    df_heading_category.columns = [' '.join(col).strip() for col in df_heading_category.columns.values]
    df_heading_category.drop(columns = ['heading_category sum'], inplace=True)
    df_heading_category.rename(columns={'heading_category_intro sum': 'heading_category_intro',
                                          'heading_category_background sum': 'heading_category_background',
                                          'heading_category_theory sum': 'heading_category_theory',
                                          'heading_category_methods sum': 'heading_category_methods',
                                          'heading_category_results sum': 'heading_category_results',
                                          'heading_category_implications sum': 'heading_category_implications',
                                          'heading_category_appendix sum': 'heading_category_appendix',
                                          'heading_category_NA sum': 'heading_category_NA'}, inplace=True)

    df = pd.merge(df_keys, df_bool, on=['citation_key_lr', 'citation_key_cp'])
    df = pd.merge(df, df_else, on=['citation_key_lr', 'citation_key_cp'])
    df = pd.merge(df, df_mention_position, on=['citation_key_lr', 'citation_key_cp'])
    df = pd.merge(df, df_heading_category, on=['citation_key_lr', 'citation_key_cp'])
    return df

if __name__ == '__main__':
    CITATION = pd.read_csv('data/interim/CITATION.csv')
    FEATURE_FRAME = prepare_dataframe(CITATION)

    CITATION_summary = summarize_citation_df(CITATION)
    FEATURE_FRAME = pd.merge(FEATURE_FRAME, CITATION_summary, on=['citation_key_lr', 'citation_key_cp'])

    LR_CP = pd.read_csv('data/interim/LR_CP.csv')
    FEATURE_FRAME = pd.merge(FEATURE_FRAME, LR_CP, on=['citation_key_lr', 'citation_key_cp'])

    LR = pd.read_csv('data/interim/LR.csv')
    FEATURE_FRAME = pd.merge(FEATURE_FRAME, LR.loc[:,['citation_key_lr', 'SYN','TT','TB','RG','CRI','RA']], on='citation_key_lr')

    CP = pd.read_csv('data/interim/CP.csv')
    FEATURE_FRAME = pd.merge(FEATURE_FRAME, CP.loc[:,['citation_key_cp', 'total_references', 'total_citations']], on='citation_key_cp')
    FEATURE_FRAME['weighted_citation_count'] = FEATURE_FRAME['focal_citations'] / FEATURE_FRAME['total_citations']

    FEATURE_FRAME['USE'] = FEATURE_FRAME['NOT'] == False
    FEATURE_FRAME.drop(columns = ['context', 'NOT'], inplace=True)
    FEATURE_FRAME = FEATURE_FRAME[['citation_key_lr',
                                    'citation_key_cp',
                                    'focal_citations',
                                    'textual',
                                    'separate',
                                    'comp_sup',
                                    'prp',
                                    'pos_0',
                                    'pos_1',
                                    'pos_2',
                                    'pos_3',
                                    'pos_4',
                                    'pos_5',
                                    'sentence_popularity amin',
                                    'sentence_popularity amax',
                                    'sentence_popularity mean',
                                    'context_popularity amin',
                                    'context_popularity amax',
                                    'context_popularity mean',
                                    'sentence_density amin',
                                    'sentence_density amax',
                                    'sentence_density mean',
                                    'context_density amin',
                                    'context_density amax',
                                    'context_density mean',
                                    'position_in_sentence amin',
                                    'position_in_sentence amax',
                                    'position_in_sentence mean',
                                    'sentence_neg amin',
                                    'sentence_neg amax',
                                    'sentence_neg mean',
                                    'sentence_neu amin',
                                    'sentence_neu amax',
                                    'sentence_neu mean',
                                    'sentence_pos amin',
                                    'sentence_pos amax',
                                    'sentence_pos mean',
                                    'sentence_compound amin',
                                    'sentence_compound amax',
                                    'sentence_compound mean',
                                    'context_neg amin',
                                    'context_neg amax',
                                    'context_neg mean',
                                    'context_neu amin',
                                    'context_neu amax',
                                    'context_neu mean',
                                    'context_pos amin',
                                    'context_pos amax',
                                    'context_pos mean',
                                    'context_compound amin',
                                    'context_compound amax',
                                    'context_compound mean',
                                    'self_citation',
                                    'title_similarity',
                                    'abstract_similarity',
                                    'SYN',
                                    'TT',
                                    'TB',
                                    'RG',
                                    'CRI',
                                    'RA',
                                    'total_references',
                                    'total_citations',
                                    'weighted_citation_count',
                                    'mention_positions_10',
                                    'mention_positions_20',
                                    'mention_positions_30',
                                    'mention_positions_40',
                                    'mention_positions_50',
                                    'mention_positions_60',
                                    'mention_positions_70',
                                    'mention_positions_80',
                                    'mention_positions_90',
                                    'mention_positions_100',
                                    'heading_category_NA',
                                    'heading_category_intro',
                                    'heading_category_background',
                                    'heading_category_theory',
                                    'heading_category_methods',
                                    'heading_category_results',
                                    'heading_category_implications',
                                    'heading_category_appendix',
                                    'ref_in_title',
                                    'ref_in_heading',
                                    'ref_in_figure_description',
                                    'ref_in_table_description',
                                    'SYN_TB',
                                    'CRI_ADDR',
                                    'RG_SYN',
                                    'RG_CLOSE',
                                    'RA_CLOSE',
                                    'TB_TB',
                                    'TB_TT',
                                    'TB_RG',
                                    'TT_TT',
                                    'TT_RG',
                                    'USE']]
    FEATURE_FRAME.to_csv('data/processed/FEATURE_FRAME.csv', index=False, quoting=csv.QUOTE_ALL)
