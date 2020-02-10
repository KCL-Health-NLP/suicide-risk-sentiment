# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:56:55 2018

@author: ABittar

This script contains utility functions to extract sentiment words from the 
Pattern lexicon to prepare data for experimaents on sentiment in clinical
notes of a data set used for suicide risk assessment.
The Pattern library is written in Python 2.7 and must be run with that version.
This code is compatible with Python 2.7.

C:/Users/ABittar/AppData/Local/Continuum/anaconda3/envs/py27/Lib/site-packages/Pattern-2.6-py2.7.egg
"""

from __future__ import absolute_import, print_function
from pattern.text import Sentiment
from pattern.en import sentiment

import pandas as pd

# Data and resource directories mapped to the T and Z drives
BASE_DIR_T = 'T:/Andre Bittar/workspace/ehost-it/suicide-sentiment/'
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'
RESOURCE_DIR = BASE_DIR_T + 'resources/'


def save_sentiment_dict(remove_neutral=True):
    """
    This saves the sentiment dictionary without extended adverbs (check).
    No get additional adverbs, modify the load function in the Sentiment class of Pattern.
    """
    # this must be the path of the Pattern sentiment lexicon for English
    pin = 'C:/Users/ABittar/AppData/Local/Continuum/anaconda3/envs/py27/lib/site-packages/pattern-2.6-py2.7.egg/pattern/text/en/en-sentiment.xml'
    s = Sentiment()
    s.load(pin)
    s = dict(s)
    df_lex = pd.DataFrame.from_dict(s).T
    
    # keep only polarity value (first element in list, or 0 if null)
    for col in df_lex.columns:
        df_lex[col] = df_lex[[col]][col].apply(lambda x: x[0] if (type(x) == list or type(x) == tuple) else 0)

    # remove rows/words with 0 values
    if remove_neutral:
        df_lex = df_lex.loc[~(df_lex[None] == 0)]
        df_lex.drop('', inplace=True, axis=1)
    
    # > 0 = pos (1), <0 = neg (-1), 0 = neutral
    for col in df_lex.columns:
        df_lex[col] = df_lex[[col]][col].apply(lambda x: 1 if x > 0 else x)

    for col in df_lex.columns:
        df_lex[col] = df_lex[[col]][col].apply(lambda x: -1 if x < 0 else x)

    df_lex.rename(columns={None: 'aggregated'}, inplace=True)

    df_lex.to_pickle(RESOURCE_DIR + 'pattern_en_sentiment_full.pickle')

    return df_lex


def load_data(ctype):
    print('-- Loading data...', end='')
    df = pd.read_pickle(BASE_DIR_Z + 'data/' + ctype + '_30_text_ordinal_dates_p2.pickle')
    print('Done.')

    return df


def save_data(df, ctype):
    print('-- Saving data...', end='')
    pout = BASE_DIR_Z + 'data/' + ctype + '_30_text_ordinal_dates_pattern_p2.pickle'
    df.to_pickle(pout)
    print('Done.')
    print('Saved file:', pout)


def apply_pattern(text):
    s = sentiment(text).assessments

    polarities = {}

    for item in s:
        tokens = item[0]
        polarity = item[1]
        subjectivity = item[2]
        
        if polarity > 0:
            # positive subjectivity
            tmp = polarities.get('subj_pos', 0.0)
            polarities['subj_pos'] = tmp + subjectivity
            # positive polqrity            
            tmp = polarities.get('pos', 0.0)
            polarities['pos'] = tmp + polarity            
            # positive polarity words
            tmp = polarities.get('pos_words', [])
            tmp += tokens
            polarities['pos_words'] = tmp
        if polarity < 0:
            # negative subjectivity
            tmp = polarities.get('subj_neg', 0.0)
            polarities['subj_neg'] = tmp + subjectivity
            # negative polarity
            tmp = polarities.get('neg', 0.0)
            polarities['neg'] = tmp + polarity
            # negative polarity words
            tmp = polarities.get('neg_words', [])
            tmp += tokens
            polarities['neg_words'] = tmp

    return pd.Series(polarities)


def fillna_list(df, col):
    inds = df.loc[df[col] == 0].index

    for i in inds:
        df.at[i, col] = []

    return df


def test():
    text = 'This is not very good. But I think it is nice.'

    s = sentiment(text)
    print(s.assessments)


if __name__ == '__main__':
    if False:
        df_case = load_data('case')
        df_control = load_data('control')
    
        df_case = df_case.merge(df_case.text_case.apply(apply_pattern), left_index=True, right_index=True)
    
        df_control_patt =  df_control.text_control.apply(apply_pattern)
        df_control = df_control.merge(df_control_patt, left_index=True, right_index=True)
    
        df_case.fillna(value=0.0, inplace=True)
        df_control.fillna(value=0.0, inplace=True)

        df_case = fillna_list(df_case, 'pos_words')
        df_case = fillna_list(df_case, 'neg_words')
    
        df_control = fillna_list(df_control, 'pos_words')
        df_control = fillna_list(df_control, 'neg_words')
    
        if False:
            save_data(df_case, 'case')
            save_data(df_control, 'control')