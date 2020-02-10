# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:27:22 2019

@author: ABittar

This script contains utility functions to extract sentiment words from the 
AFINN lexicon to prepare data for experimaents on sentiment in clinical
notes of a data set used for suicide risk assessment.
"""

from afinn import Afinn
import pandas as pd

# Data and resource directories mapped to the T and Z drives
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'


def load_data(ctype):
    print('-- Loading data...', end='')
    df = pd.read_pickle(BASE_DIR_Z + 'data/' + ctype + '_30_text_ordinal_dates_p2.pickle')
    print('Done.')

    return df


def save_data(df, ctype):
    print('-- Saving data...', end='')
    pout = BASE_DIR_Z + 'data/' + ctype + '_30_text_ordinal_dates_afinn_p2.pickle'
    df.to_pickle(pout)
    print('Done.')
    print('Saved file:', pout)
    

def test():
    afinn = Afinn()

    text = "We don't like cricket, we love it! But they hate it!"

    d = get_afinn_polarity_word_scores(text, afinn)
    p = get_words_by_polarity('negative', d)
    s = get_scores_by_polarity('positive', d)
    t = get_text_score(d)
    
    print(d, p, s, t)


def get_afinn_polarity_word_scores(text, afinn):
    scores = afinn.scores_with_pattern(text)

    d = {}
    for s in scores:
        tmp = []
        if s[3] > 0:
            tmp = d.get('positive', [])
            d['positive'] = tmp
        elif s[3] < 0:
            tmp = d.get('negative', [])
            d['negative'] = tmp
        else:
            continue
        tmp.append((s[0], s[1], s[2], float(s[3])))
    
    return d


def get_words_by_polarity(polarity, d):
    return [word[0] for word in d.get(polarity, [])]


def get_scores_by_polarity(polarity, d):
    return sum([word[3] for word in d.get(polarity, [])])


def get_text_score(d):
    pos_score = get_scores_by_polarity('positive', d)
    neg_score = get_scores_by_polarity('negative', d)
    pos_words = get_words_by_polarity('positive', d)
    neg_words = get_words_by_polarity('negative', d)
    
    return [pos_score, neg_score, sum([pos_score, neg_score]), pos_words, neg_words]


if __name__ == '__main__':
    afinn = Afinn()

    df_case = load_data('case')
    df_control = load_data('control')
    cols = ['pos', 'neg', 'score', 'pos_words', 'neg_words']
    
    df_case = df_case.merge(
            pd.DataFrame(df_case.text_case.apply(lambda x: 
                    get_text_score(get_afinn_polarity_word_scores(x, afinn))).values.tolist(),
                columns=cols),
            left_index=True, right_index=True)
    
    df_control = df_control.merge(
            pd.DataFrame(df_control.text_control.apply(lambda x: 
                    get_text_score(get_afinn_polarity_word_scores(x, afinn))).values.tolist(),
                columns=cols), 
            left_index=True, right_index=True)
    
    df_case.fillna(value=0.0, inplace=True)
    df_control.fillna(value=0.0, inplace=True)
    
    save_data(df_case, 'case')
    save_data(df_control, 'control')