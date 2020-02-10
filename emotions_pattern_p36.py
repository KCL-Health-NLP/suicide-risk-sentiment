# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:13:13 2019

@author: ABittar

This script contains utility functions to extract sentiment words from the 
Pattern lexicon to prepare data for experimaents on sentiment in clinical
notes of a data set used for suicide risk assessment.
"""

import pandas as pd

# Data and resource directories mapped to the T and Z drives
BASE_DIR_T = 'T:/Andre Bittar/workspace/ehost-it/suicide-sentiment/'
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'
RESOURCE_DIR = BASE_DIR_T + 'resources/'


def load_data_pp(ctype):
    print('-- Loading data for ' + ctype + '...', end='')
    df = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/' + ctype + '_30_text_pp_lemma.pickle')
    #df = df[0:10] # TODO remember this is here for testing!
    print('Done.')

    return df


def apply_pattern(tokens, df_lex, lex_pos):
    pos_words = []
    neg_words = []
    for token in tokens:
        split = token.split('_')
        word = split[0]
        lemma = split[1]
        pos = split[2]
        
        if pos not in lex_pos:
            continue
        
        if lemma not in df_lex.index:
            continue
        
        pol = df_lex.at[lemma, pos]
        if pol == 1:
            pos_words.append(word)
        elif pol == -1:
            neg_words.append(word)
    
    return pd.Series([pos_words, neg_words], index=['pos_words', 'neg_words'])


if __name__ == '__main__':
    df_lex = pd.read_pickle(RESOURCE_DIR + 'pattern_en_sentiment_full.pickle')
    lex_pos = set([pos for pos in df_lex.columns.tolist() if pos not in ['', None]]) # the POS that are used in the Pattern lexicon
    df_case = load_data_pp('case')
    df_control = load_data_pp('control')
    
    df_case[['pos_words', 'neg_words']] = df_case.tokens_case.apply(apply_pattern, args=(df_lex, lex_pos))
    df_case.drop('tokens_case', axis=1, inplace=True)

    df_control[['pos_words', 'neg_words']] = df_control.tokens_control.apply(apply_pattern, args=(df_lex, lex_pos))
    df_control.drop('tokens_control', axis=1, inplace=True)
    
    df_case.to_pickle(BASE_DIR_Z + 'data/case_30_text_pp_pattern_lexicon.pickle')
    df_control.to_pickle(BASE_DIR_Z + 'data/control_30_text_pp_pattern_lexicon.pickle')