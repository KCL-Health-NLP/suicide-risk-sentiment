# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:45:39 2019

@author: ABittar

This script contains utility functions to extract sentiment words from the 
NRC-Emotion Lexicon (EmoLex) to prepare data for experimaents on sentiment in clinical
notes of a data set used for suicide risk assessment.
"""

import pandas as pd
import spacy

from nltk.corpus import stopwords

# Data and resource directories mapped to the T and Z drives
BASE_DIR_T = 'T:/Andre Bittar/workspace/ehost-it/suicide-sentiment/'
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'
RESOURCE_DIR = BASE_DIR_T + 'resources/'


print('-- Loading spaCy model for English...', end='')
nlp = spacy.load('en', disable=['tagger', 'parser'])
print('Done.')


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


def load_lexicon(path):
    """
    Load a lexicon from a text file - file must be in EmoLex format.    
    - For EmoLex use: 'data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    - For LIWC lexicons use: data/LIWC_2007_pos_neg_emo.txt
    - For Opinion use: data/opinion-lexicon-English/positive-words.txt
                       data/opinion-lexicon-English/negative-words.txt
    """

    print('-- Preparing lexicon...', end='')
    entries = [line.split('\t') for line in open(path, 'r').read().split('\n')
               if line != '' and '\t0' not in line]

    df_lex = pd.DataFrame(entries, columns=['word', 'emotion', 'value'])
    df_lex['value'] = df_lex.value.astype(int)
    print('Done.')
    
    return df_lex


def get_token_emos(df_lex, text, match_on):
    doc = nlp(text)
    emotion_words = {}

    for token in doc:
        if token.lemma_ not in stopwords.words('english') and \
        token.lemma_ != '-PRON-' and \
        not token.is_punct and not token.is_space:
            if match_on == 'lemma':
                emos = df_lex.loc[df_lex.word == token.lemma_].emotion
            elif match_on == 'word':
                emos = df_lex.loc[df_lex.word == token.text].emotion
            else:
                raise Exception('-- Error: invalid match_on ' + match_on + ' choose "word" or "lemma".')
            for e in emos:
                tmp = emotion_words.get(e, [])
                if match_on == 'lemma':
                    tmp.append(token.lemma_)
                if match_on == 'word':
                    tmp.append(token.text)
                emotion_words[e] = tmp
    
    return emotion_words


def quick(df_lex, tokens):
    emotion_words = {}
    for token in tokens:
        emos = df_lex.loc[df_lex.word == token].emotion
        for e in emos:
            tmp = emotion_words.get(e, [])
            tmp.append(token)
            emotion_words[e] = tmp

    return emotion_words


def test():
    df_lex = load_lexicon(RESOURCE_DIR + 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    df_case =  load_data('case')
    
    for i, row in df_case.iterrows():
        e = get_token_emos(df_lex, 'word')
        if i % 100 == 0:
            print(i, e)


if  __name__ == '__main__':
    print('-- Thiss is just  ttooo slow')