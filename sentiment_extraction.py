# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:05:11 2019

@author: ABittar

This script calculates frequency statistics for sentiment extracted from the
eHOST-IT cohort text data.

The various functions are to be run in sequential order on previously extracted
data.
    1. calculate_polarity_word_lists(): create and store lists of the positive
       and negative words that are in each of the sub-corpora.
    2. mwu(): calculates the Mann-Whitney U test for statistical significance
       of frequency differences of words between case (suicidal) and control 
       (non-suicidal) corpora.
    3. aggregate_mwu_freq_results(): aggregate word frequency statistics for 
       each sub-corpus.
    4. freq_per_word_emotion_table_global(): create a spreadsheet indicating 
       for each word whether it is present in each sub-corpus or not.
    5. generate_strip_plot(): generate a strip plot showing the positive and 
       negative words from each lexicon that are present in each sub-corpus.
    6. calculate_lexicon_coverage(): calculate lexicon coverage statistics in 
       terms of word token and type.
    7. get_top_exclusive_words(): get the top most frequent words that are
       exclusive to each of the sub-corpora.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

from collections import Counter
from nltk.corpus import stopwords
from scipy.stats import mannwhitneyu
from time import time

# TODO Make these command line arguments. Data and resource directories mapped to the T and Z drives
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'

LEXICONS = ['afinn', 'emolex', 'liwc2015', 'opinion', 'pattern', 'swn']
LEXICON_NAME_CODES = {'afinn': 'AFN', 'emolex': 'EMO', 'liwc2015': 'LWC', 'opinion': 'OPN', 'pattern': 'PAT', 'swn': 'SWN'}
WORD_REGEX = '^[A-Za-z][^ \r\t\n]*$'


def calculate_polarity_word_lists(run_id):
    """
    Create lists of the positive and negative words that are identified in each
    sub-corpus by each of the lexicons.
    """
    base_dir = BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    
    for e in LEXICONS:
        print('-- Getting matched word lists for', e)
        if e == 'pattern':
            df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_pp_pattern_lexicon.pickle')
            df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_pp_pattern_lexicon.pickle')
        elif e == 'swn':
            df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_pp_swn_excl.pickle')
            df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_pp_swn_excl.pickle')
        else:
            df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates_' + e + '_p2.pickle')
            df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates_' + e + '_p2.pickle')
        
        case_words_pos = set([item for sublist in df_case.pos_words for item in sublist])           
        control_words_pos = set([item for sublist in df_control.pos_words for item in sublist])
        case_words_neg = set([item for sublist in df_case.neg_words for item in sublist])            
        control_words_neg = set([item for sublist in df_control.neg_words for item in sublist])
        
        if e == 'pattern':
            pd.to_pickle(case_words_pos, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_pattern_lexicon.pickle')
            pd.to_pickle(control_words_pos, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_pattern_lexicon.pickle')
            pd.to_pickle(case_words_neg, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_pattern_lexicon.pickle')
            pd.to_pickle(control_words_neg, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_pattern_lexicon.pickle')
        else:
            pd.to_pickle(case_words_pos, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + e + '.pickle')
            pd.to_pickle(control_words_pos, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + e + '.pickle')
            pd.to_pickle(case_words_neg, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_' + e + '.pickle')
            pd.to_pickle(control_words_neg, BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_' + e + '.pickle')


def load_corpus(run_id, case=True, control=True, reload=False, use_medinfo_train=True):
    """
    Load the case and/or control sub-corpora
    """
    # remove stopwords
    sw = stopwords.words('english')
    
    words_case = []
    words_control = []

    base_dir = BASE_DIR_Z + 'data/clpsych/corpus/' + run_id
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    if use_medinfo_train:
        df_train = pd.read_pickle(BASE_DIR_Z + 'data/backtracking/df_cc_struct_with_train_pks.pickle')

    if case:
        if reload:
            df_case = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/case_30_text_pp.pickle')
            
            if use_medinfo_train:
                df_case = df_case.loc[df_case.pk.isin(df_train.pk)]

            print('-- Using', len(df_case), 'case documents')
            
            df_case['tokens'] = df_case.tokens_case.apply(lambda x: [token.split('_')[0] for token in x])
            words_case = [item for sublist in df_case.tokens.tolist() for item in sublist]
            words_case = [w for w in words_case if re.search(WORD_REGEX, w) is not None and w not in sw]
            pout = BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_case_words.pickle'
            pd.to_pickle(words_case, pout)
            print('-- Saved case words to', pout)
        else:
            words_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_case_words.pickle')
    if control:
        if reload:
            df_control = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/control_30_text_pp.pickle')

            if use_medinfo_train:
                df_control = df_control.loc[df_control.pk.isin(df_train.pk)]

            print('-- Using', len(df_control), 'control documents')

            df_control['tokens'] = df_control.tokens_control.apply(lambda x: [token.split('_')[0] for token in x])
            words_control = [item for sublist in df_control.tokens.tolist() for item in sublist]
            #words_control = [w for w in words_control if re.search(regex, w) is None and w not in sw]
            words_control = [w for w in words_control if re.search(WORD_REGEX, w) is not None and w not in sw]
            pout = BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_control_words.pickle'
            pd.to_pickle(words_control, pout)
            print('-- Saved control words to', pout)
        else:
            words_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_control_words.pickle')
    
    if case and control and reload:
            pout = BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/set_common_words.pickle'
            common_words = set(words_case).intersection(set(words_control))
            pd.to_pickle(common_words, pout)
            print('-- Saved common words to', pout)
    
    return words_case, words_control


def get_top_exclusive_words():
    """
    Get the top-ranking exclusive words for each of the sub-corpora,
    along with their frequency.
    """
    words_case, words_control = load_corpus(run_id, case=True, control=True, reload=False, use_medinfo_train=True)
    words_common = set(words_case).intersection(set(words_control))

    freq_case = Counter(words_case)
    freq_control = Counter(words_control)
    
    # calculate relative frequencies
    total_case = sum(freq_case.values())
    total_control = sum(freq_control.values())
    
    for w in freq_case:
        freq_case[w] = freq_case[w] * 1000000 / total_case
    
    for w in freq_control:
        freq_control[w] = freq_control[w] * 1000000 / total_control

    # remove common words
    for w in words_common:
        freq_case.pop(w)
        freq_control.pop(w)
    
    top_excl_case_word = freq_case.most_common(1)
    top_excl_control_word = freq_control.most_common(1)
    
    return freq_case, freq_control, top_excl_case_word, top_excl_control_word


def count_words(run_id, verbose=True):
    """
    Collect word-level stats from the preprocessed corpus files.
    """
    words_case, words_control = load_corpus(run_id, case=True, control=True, reload=False, use_medinfo_train=True)
    n_case = len(words_case)
    n_case_unique = len(set(words_case))
    n_control = len(words_control)
    n_control_unique = len(set(words_control))
    tt_case = len(set(words_case)) / len(words_case) * 100
    tt_control = len(set(words_control)) / len(words_control) * 100
    
    if verbose:
        print('Case words (token)      :', n_case)
        print('Case words (type)       :', n_case_unique)
        print('Control words (token)   :', n_control)
        print('Control words (type)    :', n_control_unique)
        print('Case token-type ratio   :', tt_case)
        print('Control token-type ratio:', tt_control)
    
    return n_case, n_case_unique, n_control, n_control_unique, tt_case, tt_control


def calculate_lexicon_coverage(run_id):
    """
    Calculates coverage of word tokens and word types
    PLOS ONE: Table 5
    """
    print('Calculating lexicon coverage...')
    s_case_pos = s_case_neg = s_control_pos = s_control_neg = None
    df_case = df_control = None
    
    # needs to be run with a non-blank run_id of a run that uses freq_diff
    # also must include non-significant words - ie don't filter by p-value before doing this

    # count corpus word tokens
    l_case_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_case_words.pickle')
    l_control_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_control_words.pickle')

    n_case = len(l_case_words)
    n_control = len(l_control_words)
    n_total = n_case + n_control

    print('-- Calculating for: corpus')
    print('Total case words (tokens)    :', n_case)
    print('Total control words (tokens) :', n_control)

    # count corpus word types
    s_case_words = set(l_case_words)
    s_control_words = set(l_control_words)
    
    n_case_unique = len(s_case_words)
    n_control_unique = len(s_control_words)
    n_total_unique = len(s_case_words.union(s_control_words))

    print('Total unique case words   :', n_case_unique)
    print('Total unique control words:', n_control_unique)
    
    for e in LEXICONS:
        print('-- Calculating for:', e)

        # load this to count word tokens
        df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates_' + e + '_p2.pickle')
        df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates_' + e + '_p2.pickle')

        if e == 'pattern':
            # these are sets of words - the word types for each lexicon
            s_case_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_pattern_lexicon.pickle')
            s_case_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_pattern_lexicon.pickle')
            s_control_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_pattern_lexicon.pickle')
            s_control_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_pattern_lexicon.pickle')
        else:
            s_case_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + e + '.pickle')
            s_case_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_' + e + '.pickle')
            s_control_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + e + '.pickle')
            s_control_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_' + e + '.pickle')
        
        # count tokens
        l_case_tokens = len([word for sublist in df_case.pos_words.tolist() for word in sublist]) + len([word for sublist in df_case.neg_words.tolist() for word in sublist])
        l_control_tokens = len([word for sublist in df_control.pos_words.tolist() for word in sublist]) + len([word for sublist in df_control.neg_words.tolist() for word in sublist])
        l_total_tokens = l_case_tokens + l_control_tokens
        
        lex_case_words = len(s_case_neg.union(s_case_pos))
        lex_control_words = len(s_control_neg.union(s_control_pos))
        lex_total_words = len(s_control_neg.union(s_control_pos).union(s_case_neg.union(s_case_pos)))
        
        print('-----')
        print(e)
        print('-----')

        case_pc_tokens = round(l_case_tokens / n_case * 100, 1)
        control_pc_tokens = round(l_control_tokens / n_control * 100, 1)
        total_pc_tokens = round((l_case_tokens + l_control_tokens) / (n_case + n_control) * 100, 1)
        
        case_pc = round(lex_case_words / n_case_unique * 100, 1)
        control_pc = round(lex_control_words / n_control_unique * 100, 1)
        total_pc = round((lex_case_words + lex_control_words) / (n_case_unique + n_control_unique) * 100, 1)

        print('Word tokens:')
        print('Case coverage   :', l_case_tokens, n_case, '(' + str(case_pc_tokens) + '%)')
        print('Control coverage:', l_control_tokens, n_control, '(' + str(control_pc_tokens) + '%)')
        print('Total coverage  :', l_total_tokens, n_total, '(' + str(total_pc_tokens) + '%)')
        print('-----')
        print('Word types:')
        print('Case coverage   :', lex_case_words, n_case_unique, '(' + str(case_pc) + '%)')
        print('Control coverage:', lex_control_words, n_control_unique, '(' + str(control_pc) + '%)')
        print('Total coverage  :', lex_total_words, n_total_unique, '(' + str(total_pc) + '%)')
            
    print('Done.')


def calculate_sentiment_coverage_global_tokens(run_id):
    """
    Calculate percentage coverage over all actual tokens, as opposed to just types,
    as done by calculate_sentiment_coverage_global.
    Maybe do both case and control in here...
    """
    
    df = pd.DataFrame(columns=LEXICONS)
    
    # not most efficient...
    l_words_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_case_words.pickle')
    l_words_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/list_control_words.pickle')

    c_case_pos = Counter(l_words_case)
    c_case_neg = Counter(l_words_case)
    c_control_pos = Counter(l_words_control)
    c_control_neg = Counter(l_words_control)

    for e in LEXICONS:
        # positive
        if e == 'pattern':
            pos_lex_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + e + '_lexicon.pickle')
            pos_lex_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + e + '_lexicon.pickle')
        else:
            pos_lex_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + e + '.pickle')
            pos_lex_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + e + '.pickle')

        pos_diff_case = set(c_case_pos.keys()).difference(pos_lex_case)
        for w in pos_diff_case:
            c_case_pos.pop(w)
        pos_lex_total_case = sum(c_case_pos.values())

        pos_diff_control = set(c_control_pos.keys()).difference(pos_lex_control)
        for w in pos_diff_control:
            c_control_pos.pop(w)
        pos_lex_total_control = sum(c_control_pos.values())

        # negative
        if e == 'pattern':
            neg_lex_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_' + e + '_lexicon.pickle')
            neg_lex_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_' + e + '_lexicon.pickle')
        else:
            neg_lex_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_' + e + '.pickle')
            neg_lex_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_' + e + '.pickle')

        neg_diff_case = set(c_case_neg.keys()).difference(neg_lex_case)
        for w in neg_diff_case:
            c_case_neg.pop(w)
        neg_lex_total_case = sum(c_case_neg.values())

        neg_diff_control = set(c_control_neg.keys()).difference(neg_lex_control)
        for w in neg_diff_control:
            c_control_neg.pop(w)
        neg_lex_total_control = sum(c_control_neg.values())

        lex_total_case = pos_lex_total_case + neg_lex_total_case
        pos_lex_total_case / lex_total_case * 100
        neg_lex_total_case / lex_total_case * 100

        lex_total_control = pos_lex_total_control + neg_lex_total_control
        pos_lex_total_control / lex_total_control * 100
        neg_lex_total_control / lex_total_control * 100

        df.at[0, e] = pos_lex_total_case
        df.at[1, e] = neg_lex_total_case
        df.at[2, e] = lex_total_case
        df.at[3, e] = pos_lex_total_case / lex_total_case
        df.at[4, e] = neg_lex_total_case / lex_total_case

        df.at[5, e] = pos_lex_total_control
        df.at[6, e] = neg_lex_total_control
        df.at[7, e] = lex_total_control
        df.at[8, e] = pos_lex_total_control / lex_total_control
        df.at[9, e] = neg_lex_total_control / lex_total_control
        
        print('-----')
        print(e)
        print('-----')
    
        print('Positive coverage case   :', pos_lex_total_case, '/', lex_total_case, pos_lex_total_case / lex_total_case * 100, '%')
        print('Positive coverage control:', pos_lex_total_control, '/', lex_total_control, pos_lex_total_control / lex_total_control * 100, '%')
        print('Negative coverage case   :', neg_lex_total_case, '/', lex_total_case, neg_lex_total_case / lex_total_case * 100, '%')
        print('Negative coverage control:', neg_lex_total_control, '/', lex_total_control, neg_lex_total_control / lex_total_control * 100, '%')

    df.index = ['n_pos_case', 'n_neg_case', 'n_total_case', '%_pos_case', '%_neg_case',
                'n_pos_control', 'n_neg_control', 'n_total_control', '%_pos_control', '%_neg_control']
    
    pout = BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/sentiment_coverage_global_tokens.xlsx'
    print('-- Writing file:', pout)
    
    df.to_excel(pout)

    return df


def calculate_sentiment_coverage_global_types(run_id):
    print('Calculating global sentiment coverage...')

    df = pd.DataFrame(columns=LEXICONS)

    for e in LEXICONS:
        print('-- Calculating for:', e)
        if e == 'pattern':
            # these are not actually DataFrames, but sets of words
            s_case_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_pattern_lexicon.pickle')
            s_case_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_pattern_lexicon.pickle')
            s_control_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_pattern_lexicon.pickle')
            s_control_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_pattern_lexicon.pickle')
        else:
            s_case_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + e + '.pickle')
            s_case_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_' + e + '.pickle')
            s_control_pos = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + e + '.pickle')
            s_control_neg = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_neg_words_' + e + '.pickle')
        
        lex_pos_case = len(s_case_pos)
        lex_pos_control = len(s_control_pos)
        
        lex_neg_case = len(s_case_neg)
        lex_neg_control = len(s_control_neg)
        
        lex_total_case = lex_pos_case + lex_neg_case
        lex_total_control = lex_pos_control + lex_neg_control
        
        print('-----')
        print(e)
        print('-----')
    
        print('Positive coverage case   :', lex_pos_case, '/', lex_total_case, lex_pos_case / lex_total_case * 100, '%')
        print('Positive coverage control:', lex_pos_control, '/', lex_total_control, lex_pos_control / lex_total_control * 100, '%')
        print('Negative coverage case   :', lex_neg_case, '/', lex_total_case, lex_neg_case / lex_total_case * 100, '%')
        print('Negative coverage control:', lex_neg_control, '/', lex_total_control, lex_neg_control / lex_total_control * 100, '%')

        df.at[0, e] = lex_pos_case
        df.at[1, e] = lex_neg_case
        df.at[2, e] = lex_total_case
        df.at[3, e] = lex_pos_case / lex_total_case
        df.at[4, e] = lex_neg_case / lex_total_case

        df.at[5, e] = lex_pos_control
        df.at[6, e] = lex_neg_control
        df.at[7, e] = lex_total_control
        df.at[8, e] = lex_pos_control / lex_total_control
        df.at[9, e] = lex_neg_control / lex_total_control

    print('Done.')
    
    df.index = ['n_pos_case', 'n_neg_case', 'n_total_case', '%_pos_case', '%_neg_case',
                'n_pos_control', 'n_neg_control', 'n_total_control', '%_pos_control', '%_neg_control']
    
    pout = BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/sentiment_coverage_global_types.xlsx'
    print('-- Writing file:', pout)
    
    df.to_excel(pout)
    
    return df


def test_sentiment_proportion_significance(run_id):
    """
    This is not used in the paper
    """
    mwu_scores = {}
    base_dir = BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id

    # randomly sample n documents from each population
    for e in LEXICONS:
        print('-- Loading data for', e + '...', end='')
        if e == 'pattern':
            df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_pp_pattern_lexicon.pickle')
            df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_pp_pattern_lexicon.pickle')
        elif e == 'swn':
            df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_pp_swn_excl.pickle')
            df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_pp_swn_excl.pickle')
        else:
            df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates_' + e + '_p2.pickle')
            df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates_' + e + '_p2.pickle')

        print('Done.')

        # calculate proportions of positive and negative sentiment per document
        df_pos_case = df_case.pos_words.apply(lambda x: len(x))
        df_neg_case = df_case.neg_words.apply(lambda x: len(x))
        
        df_pos_case = df_pos_case / (df_pos_case + df_neg_case)
        df_neg_case = df_neg_case / (df_pos_case + df_neg_case)
        
        df_pos_case.fillna(value=0, inplace=True)
        df_neg_case.fillna(value=0, inplace=True)

        df_pos_control = df_control.pos_words.apply(lambda x: len(x))
        df_neg_control = df_control.neg_words.apply(lambda x: len(x))

        df_pos_control = df_pos_control / (df_pos_control + df_neg_control)
        df_neg_control = df_neg_control / (df_pos_control + df_neg_control)

        df_pos_control.fillna(value=0, inplace=True)
        df_neg_control.fillna(value=0, inplace=True)
        
        print('-- Calculating MWU...')

        mwu_pos = mannwhitneyu(df_pos_case.tolist(), df_pos_control.tolist())
        mwu_neg = mannwhitneyu(df_neg_case.tolist(), df_neg_control.tolist())
        print(e, mwu_pos, mwu_neg)
        mwu_scores[e] = {'pos': mwu_pos, 'neg': mwu_neg}
        
    pd.to_pickle(mwu_scores, os.path.join(base_dir, 'mwu_sentiment_proportions_all.pickle'))
    return mwu_scores


def mwu(run_id, reload, use_medinfo_train):
    """
    Test significance of word frequencies between the two sub-corpora using the
    Mann-Whitney (Wilcoxon) U-test.
    This uses the set of common words calculated with load_corpus, which 
    contains the word definition as regex and stopword list.
    PLOS ONE: Table 4
    """
    t0 = time()
    
    print('Calculating Mann-Whitney U-Test scores')
    
    # Calculate word lists and common words
    if reload:
        _, _ = load_corpus(run_id, case=True, control=True, reload=True, use_medinfo_train=True)
    
    sw = stopwords.words('english')
    
    print('-- Load case pre-processed text...', end='')

    df_case_pp = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/case_30_text_pp.pickle')

    if use_medinfo_train:
        df_train = pd.read_pickle(BASE_DIR_Z + 'data/backtracking/df_cc_struct_with_train_pks.pickle')
        df_case_pp = df_case_pp.loc[df_case_pp.pk.isin(df_train.pk)]

    print('-- Using', len(df_case_pp), 'case documents')

    df_case_pp['words_case'] = df_case_pp.tokens_case.apply(lambda x: [y.split('_')[0] for y in x])
    df_case_pp['words_case'] = df_case_pp.words_case.apply(lambda x: [w for w in x if re.search(WORD_REGEX, w) is not None and w not in sw])
       
    print('Done.')
    
    case_doc_word_counts =  []
    
    print('-- Collecting words stats for all case documents...', end='')
    
    for i, row in df_case_pp.iterrows():
        counts = Counter(row.words_case)
        case_doc_word_counts.append(counts)
        if i % 1000 ==  0:
            print(i)
    
    for c in case_doc_word_counts:
        total = sum(c.values())
        for w in c:
            # calculate normalised frequencies (wpm)
            c[w] = (c[w] * 1000000) / total
    
    print('Done.')
    
    # load the set of common words
    cc_words_common = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/set_common_words.pickle')
    #cc_words_common = set([w for w in cc_words_common if re.search(WORD_REGEX, w) is not None and w not in sw])
    
    print('-- Load control pre-processed text...', end='')
    
    df_control_pp = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/control_30_text_pp.pickle')

    if use_medinfo_train:
        df_control_pp = df_control_pp.loc[df_control_pp.pk.isin(df_train.pk)]

    print('-- Using', len(df_control_pp), 'control documents')

    df_control_pp['words_control'] = df_control_pp.tokens_control.apply(lambda x: [y.split('_')[0] for y in x])
    df_control_pp['words_control'] = df_control_pp.words_control.apply(lambda x: [w for w in x if re.search(WORD_REGEX, w) is not None and w not in sw])

    print('Done.')
    
    control_doc_word_counts =  []
    
    print('-- Collecting words stats for all control documents...', end='')
    
    for i, row in df_control_pp.iterrows():
        counts = Counter(row.words_control)
        control_doc_word_counts.append(counts)
        if i % 1000 ==  0:
            print(i)
    
    for c in control_doc_word_counts:
        total = sum(c.values())
        for w in c:
            # calculate normalised frequencies (wpm)
            c[w] = (c[w] * 1000000) / total
    
    print('Done.')
    
    mwu_results = {}
    
    n = len(cc_words_common)
    print('Calculating Mann-Whitney U for', n, 'words...')
    
    i = 0
    for word in cc_words_common:
        case_word_freqs = [c[word] for c in case_doc_word_counts]
        control_word_freqs = [c[word] for c in control_doc_word_counts]
        result = mannwhitneyu(case_word_freqs, control_word_freqs)
        mwu_results[word] = result
        if i % 1000 == 0:
            print(i, '/', n)
        i += 1
    
    print('Done.')
    
    pd.to_pickle(case_doc_word_counts, BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/case_doc_wcounts.pickle')
    pd.to_pickle(control_doc_word_counts, BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/control_doc_wcounts.pickle')
    pd.to_pickle(mwu_results, BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/mwu_results_wpm.pickle')

    t1 = time()
    print('Time:', t1 - t0)
    
    return case_doc_word_counts, control_doc_word_counts, mwu_results


def aggregate_mwu_freq_results(mwu_results, run_id, reload=False):
    """
    -- Calculate sub-corpus-wide word frequencies
    -- Calculate word frequency ratio
    -- Create new DataFrame with all information for all common words
    PLOS ONE: Table 4
    """
    
    print('Aggregating MWU and frequency results...')

    words_case, words_control = load_corpus(run_id, case=True, control=True, reload=reload, use_medinfo_train=True)
    case_total_wc, _, control_total_wc, _, _, _ = count_words(run_id, verbose=False)
    
    # get frequencies across corpus
    # for cases:
    # - load counts from file
    print('-- Calculating frequencies for case words...', end='')
    # - add all counts and divide by total number of corpus tokens, calculate as wpm
    case_corpus_word_counts = Counter(words_case) # save this as it takes a long time
    for word in case_corpus_word_counts:
        case_corpus_word_counts[word] = case_corpus_word_counts[word] * 1000000 / case_total_wc
    print('Done.')

    # for controls:
    # - load counts from file
    print('-- Calculating frequencies for control words...', end='')
    # - add all counts and divide by total number of corpus tokens, calculate as wpm
    control_corpus_word_counts = Counter(words_control) # save this as it takes a long time
    for word in control_corpus_word_counts:
        control_corpus_word_counts[word] = control_corpus_word_counts[word] * 1000000 / control_total_wc
    print('Done.')

    # for common words:
    print('-- Collecting frequencies and MWU results...', end='')
    #common_words = set(case_corpus_word_counts.keys()).intersection(set(control_corpus_word_counts.keys()))
    common_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/corpus/' + run_id + '/set_common_words.pickle')
    # - store word and case_freq and control_freq in dataframe
    corpus_word_freqs = []
    for word in common_words:
        u = mwu_results[word].statistic
        p = mwu_results[word].pvalue
        case_corpus_freq = case_corpus_word_counts[word]
        control_corpus_freq = control_corpus_word_counts[word]
        corpus_word_freqs.append([word, case_corpus_freq, control_corpus_freq, u, p])
    df_corpus_word_freqs = pd.DataFrame(corpus_word_freqs, columns=['word', 'case_freq', 'control_freq', 'mwu_u', 'mwu_p'])
    print('Done.')

    # - calculate freq-ratio case-control, control-case
    print('-- Calculating frequency ratio...', end='')
    df_corpus_word_freqs['freq_ratio'] = df_corpus_word_freqs.case_freq / df_corpus_word_freqs.control_freq
    print('Done.')

    print('-- Calculating frequency difference...', end='')
    df_corpus_word_freqs['freq_diff'] = df_corpus_word_freqs.case_freq - df_corpus_word_freqs.control_freq
    print('Done.')

    # now add mwu values
    
    pout = BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/cc_freqs_mwu_results_wpm.pickle'
    df_corpus_word_freqs.to_pickle(pout)
    
    print('Saved file:', pout)

    return df_corpus_word_freqs


def get_ctype_words_per_lexicon(ctype, e, run_id):
    """
    Get a DataFrame with all words matched by the given lexicon, for the given sub-corpus.
    """
    if e == 'pattern':
        neg_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/' + ctype + '_neg_words_pattern_lexicon.pickle')
        pos_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/'  + ctype + '_pos_words_pattern_lexicon.pickle')
    else:
        neg_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/'  + ctype + '_neg_words_' + e + '.pickle')
        pos_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/'  + ctype + '_pos_words_' + e + '.pickle')
    
    df = pd.DataFrame(sorted(neg_words.union(pos_words)), columns=['word_' + e])

    return df


def freq_per_word_emotion_table_global(ctype, run_id, assign='index', p_threshold=0.000001):
    """
    Generate an Excel spreadsheet of words and frequencies (i.e. in the entire corpus) and mark
    words present in each lexicon with the index of the word.
    e.g.
    
    word	case_freq	control_freq	mwu_u	mwu_p	freq_ratio	freq_diff	afinn	emolex	opinion	pattern	swn
    0	zzzzz	32795.28862	35788.81356	9894316407	7.36147E-24	0.916355848	-2993.524945	0	0	0	0	0
    1	mental	2797.915719	3825.901704	9559573748	2.244E-249	0.731308835	-1027.985985	0	0	0	1	0
    2	appointment	1440.101199	2427.490569	9668436929	1.1381E-236	0.593246877	-987.3893694	0	0	0	0	0
    ...
    10	dear	508.727278	977.1570486	9648184162	0	0.52061977	-468.4297706	10	10	0	0	10
    11	next	1214.403002	1665.763781	9795957708	1.3288E-124	0.729036743	-451.3607791	0	0	0	0	0
    12	ms	755.1657655	1202.551009	9999872208	9.32239E-60	0.627969841	-447.3852432	0	0	0	0	12
    13	stable	388.1638031	812.934644	9708713719	0	0.477484637	-424.7708409	13	13	13	0	0
    14	sincerely	341.2873751	760.4565153	9676453440	0	0.448792756	-419.1691402	14	0	14	14	14
    ...
    """
    assert ctype in ['case', 'control']

    def load_word_dict(emo_source):
        """
        Create a dictionary of all words matched in the texts to their polarity class
        """
        if e == 'pattern':
            # these are not actually DataFrames, but sets of words
            pos_w = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_pattern_lexicon.pickle')
            pos_w = pos_w.union(pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_pattern_lexicon.pickle'))

            neg_w = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_pattern_lexicon.pickle')
            neg_w = neg_w.union(pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id +'/control_neg_words_pattern_lexicon.pickle'))

        else:
            pos_w = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + emo_source + '.pickle')
            pos_w = pos_w.union(pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + emo_source + '.pickle'))
        
            neg_w = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_neg_words_' + emo_source + '.pickle')
            neg_w = neg_w.union(pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id +'/control_neg_words_' + emo_source + '.pickle'))

        d = {}
        for w in pos_w:
            d[w] = 'positive'

        for w in neg_w:
            d[w] = 'negative'

        return d
    
    df_main = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/cc_freqs_mwu_results_wpm.pickle')

    # remove non-significant words
    df_main = df_main.loc[df_main.mwu_p < p_threshold]

    if ctype == 'case':
        df_main.sort_values(by='freq_diff', ascending=False, inplace=True)

    if ctype == 'control':
        df_main.sort_values(by='freq_diff', ascending=True, inplace=True)

    df_main.reset_index(drop=True, inplace=True)

    for e in LEXICONS:
        df_w = get_ctype_words_per_lexicon('case', e, run_id)
        df_w.append(get_ctype_words_per_lexicon('control', e, run_id))
        df_w.drop_duplicates(inplace=True)

        if assign == 'category':
            d = load_word_dict(e)
            df_main[e] = df_main.word.apply(lambda x: d[x] if x in df_w['word_' + e].tolist() else '')
        elif assign == 'index':
            e_word_list = df_w['word_' + e].tolist()
            for i, row in df_main.iterrows():
                if row.word in e_word_list:
                    df_main.at[i, e] = i
                else:
                    df_main.at[i, e] = 0

    pout = BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/mwu_' + ctype + '_table_global_' + assign + '_freq_diff.xlsx'
    df_main.to_excel(pout)
    print('-- Wrote global file:', pout)

    return df_main


def generate_strip_plot(run_id):
    """
    Generate the final strip plot of the 100 top-ranked words indicating their
    presence/absence in each of the lexicons as well as the word's polarity in
    that lexicon.
    """
    df_main_case_top100 = freq_per_word_emotion_table_global('case', run_id)#[0:100]
    df_main_control_top100 = freq_per_word_emotion_table_global('control', run_id)#[0:100]
    
    df_main_case_top100 = df_main_case_top100[['word'] + LEXICONS]
    new_df = pd.DataFrame(columns=['Lexicon', 'Word Rank'])
    
    for e in df_main_case_top100.columns:
        if e != 'word':
            if e == 'pattern':
                pos_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + e + '_lexicon.pickle')
            else:
                pos_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/case_pos_words_' + e + '.pickle')
        else:
            pos_words = []
        tmp = pd.DataFrame()
        tmp['Word Rank'] = df_main_case_top100[e]
        tmp['Lexicon'] = LEXICON_NAME_CODES.get(e, 'word')
        tmp['Sub-corpus'] = 'case'
        tmp['Sentiment'] = df_main_case_top100.word.apply(lambda x: 'positive' if x in pos_words else 'negative')
        new_df = new_df.append(tmp)
    
    df_main_control_top100 = df_main_control_top100[['word'] + LEXICONS]
    new_df2 = pd.DataFrame(columns=['Lexicon', 'Word Rank'])
    
    for e in df_main_control_top100.columns:
        if e != 'word':
            if e == 'pattern':
                pos_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + e + '_lexicon.pickle')
            else:
                pos_words = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/polarity_word_lists/' + run_id + '/control_pos_words_' + e + '.pickle')
        else:
            pos_words = []
        tmp = pd.DataFrame()
        tmp['Word Rank'] = df_main_control_top100[e]
        tmp['Lexicon'] = LEXICON_NAME_CODES.get(e, 'word')
        tmp['Sub-corpus'] = 'control'
        tmp['Sentiment'] = df_main_control_top100.word.apply(lambda x: 'positive' if x in pos_words else 'negative')
        new_df2 = new_df2.append(tmp)

    new_df = new_df.append(new_df2)
    
    # now remove the words
    new_df = new_df.loc[new_df.Lexicon != 'word']
    new_df = new_df.loc[new_df['Word Rank'] != 0]
    
    new_df_pos = new_df.loc[new_df.Sentiment == 'positive']
    new_df_neg = new_df.loc[new_df.Sentiment == 'negative']
    
    # Important: all lexicons need to have at least one value, or else the plot
    # shuffles to fill the empty space
    # Filling values of -10 so they don't show on the plot
    for ln in LEXICON_NAME_CODES:
        code = LEXICON_NAME_CODES.get(ln)
        if code != 'word' and code not in new_df_pos.Lexicon.tolist():
            df1 = pd.DataFrame.from_dict({'Lexicon': [code], 'Sentiment': ['positive'], 'Sub-corpus': ['case'], 'Word Rank': [-10]})
            df2 = pd.DataFrame.from_dict({'Lexicon': [code], 'Sentiment': ['positive'], 'Sub-corpus': ['control'], 'Word Rank': [-10]})
            new_df_pos = new_df_pos.append(df1)
            new_df_pos = new_df_pos.append(df2)
            new_df_pos.sort_values(by=['Sub-corpus', 'Lexicon'], inplace=True)
        if code not in new_df_neg.Lexicon.tolist():
            df1 = pd.DataFrame.from_dict({'Lexicon': [code], 'Sentiment': ['negative'], 'Sub-corpus': ['case'], 'Word Rank': [-10]})
            df2 = pd.DataFrame.from_dict({'Lexicon': [code], 'Sentiment': ['negative'], 'Sub-corpus': ['control'], 'Word Rank': [-10]})
            new_df_neg = new_df_neg.append(df1)
            new_df_neg = new_df_neg.append(df2)
            new_df_neg.sort_values(by=['Sub-corpus', 'Lexicon'], inplace=True)

    font = {'family' : 'Arial',
            'size'   : 12}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    ax.grid(True)

    sns.stripplot(x=new_df_neg['Lexicon'], y=new_df_neg['Word Rank'], 
                  hue=new_df_neg['Sub-corpus'], jitter=False, dodge=True, 
                  size=8, ax=ax, marker='s',
                  palette={'case': 'xkcd:red', 'control': 'xkcd:red'})

    plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
   
    ax.set_xlabel("Lexicon", fontsize=20)
    ax.set_ylabel("Word Rank", fontsize=20)
    ax.set_ylim(0, 105)

    major_ticks = np.arange(0, 105, 10)
    ax.set_yticks(major_ticks)
    ax.tick_params(labelsize=16)
    
    ax2 = ax.twinx()
    
    sns.stripplot(x=new_df_pos['Lexicon'], y=new_df_pos['Word Rank'], 
                  hue=new_df_pos['Sub-corpus'], jitter=False, dodge=True, 
                  size=8, ax=ax2, marker='o',
                  palette={'case': 'xkcd:green', 'control': 'xkcd:green'})

    ax.legend_.remove()
    ax2.legend_.remove()
    
    # set plot background colour - very light grey
    ax.set_facecolor((0.95, 0.95, 0.95))
    ax2.set_facecolor((0.95, 0.95, 0.95))
    
    ax2.set_xlabel("", fontsize=20)
    ax2.set_ylabel("", fontsize=20)
    ax2.set_ylim(0, 105)
    ax2.set_yticks(major_ticks)
    
    ax2.tick_params(labelsize=16)

    pout = BASE_DIR_Z + 'data/clpsych/mwu_results/' + run_id + '/cc_mwu_top100_coverage_freq_diff.png'
    plt.tight_layout()
    plt.savefig(pout, dpi=300)

    return ax, new_df_pos, new_df_neg


if __name__ == '__main__':
    run_id = '20190418'
    run_id = '20190712'

    if False:
        #calculate_polarity_word_lists(run_id)
        case_doc_word_counts, control_doc_word_counts, mwu_results = mwu(run_id, False, True)
        df_corpus_word_freqs = aggregate_mwu_freq_results(mwu_results, run_id)
        df_main_case = freq_per_word_emotion_table_global('case', run_id, assign='category')
        df_main_control = freq_per_word_emotion_table_global('control', run_id, assign='category')
        ax, new_df_pos, new_df_neg = generate_strip_plot(run_id)
