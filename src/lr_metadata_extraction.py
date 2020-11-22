#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

if __name__ == "__main__":
    LR_CP = pd.read_csv('data/raw/LR_CP.csv')
    LR = pd.read_csv('data/raw/LR.csv')
    LR = LR.loc[LR['BV'] == True]
    LR_CP = LR_CP[['citation_key_lr']].drop_duplicates()
    LR = pd.merge(LR_CP, LR, how='left')
    LR = LR[['citation_key_lr',
             'title',
             'synthesis', 
             'theory_testing', 
             'theory_building', 
             'r_gaps', 
             'criticizing', 
             'r_agenda',
             'gs_citations_2016_07', 
             'abstract']]
    LR.rename(columns={'synthesis': 'SYN', 
                        'theory_testing': 'TT', 
                        'theory_building': 'TB', 
                        'r_gaps': 'RG', 
                        'criticizing': 'CRI', 
                        'r_agenda': 'RA',
                        'gs_citations_2016_07': 'citation_count'}, inplace=True)

    LR[['SYN', 'TT', 'TB', 'RG', 'CRI', 'RA']] = LR[['SYN', 'TT', 'TB', 'RG', 'CRI', 'RA']].notnull().astype('bool')
    LR.to_csv('data/interim/LR.csv', index=False)
