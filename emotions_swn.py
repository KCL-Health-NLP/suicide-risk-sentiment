# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:02:38 2019

@author: ABittar

This script contains utility functions to extract sentiment words from the 
SentiWordNet lexicon to prepare data for experimaents on sentiment in clinical
notes of a data set used for suicide risk assessment.

"""

import pandas as pd
import random
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader import WordNetError

# Data and resource directories mapped to the T and Z drives
BASE_DIR_T = 'T:/Andre Bittar/workspace/ehost-it/suicide-sentiment/'
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'
RESOURCE_DIR = BASE_DIR_T + 'resources/'


POS_MAP = {'ADJ': 'a',
         'NOUN': 'n',
         'ADV': 'r',
         'VERB': 'v'
         }


nlp = None#spacy.load('en')


def test(text):
    sentiments = {}
    for token in nlp(text):
        print(token.lemma_, token.pos_)
        swn_pos = POS_MAP.get(token.pos_, None)
        if swn_pos is None:
            continue
        query = token.lemma_ + '.' + swn_pos + '.01'
        swn_syn = swn.senti_synset(query)
        print(swn_syn)
        pos = swn_syn.pos_score()
        neg = swn_syn.neg_score()
        if pos == 0 and neg == 0:
            continue
        
        if pos > 0:
            tmp = sentiments.get('positive', [])
            tmp.append((token.lower_, token.i, token.i + len(token.text), pos))
            sentiments['positive'] = tmp
        
        if neg > 0:
            tmp = sentiments.get('negative', [])
            tmp.append((token.lower_, token.i, token.i + len(token.text), -neg))
            sentiments['negative'] = tmp
    
    return sentiments


def test_2():
    t1 = 'These are the good test of the greatly awaited spacy programme. :-)'
    t2 = 'This is the worst thing and I feel horrible, but it is nice and happy.'

    texts = []
    texts.append(t1)
    texts.append(t2)

    df = pd.DataFrame(columns=['neg', 'neg_words', 'pos', 'pos_words'])
    for i in range(len(texts)):
        t = texts[i]
        tokens = [token.lower_ + '_' + token.lemma_ + '_' + POS_MAP.get(token.pos_, 'UNK') for token in nlp(t)]
    
        #s = test(text)

        from pprint import pprint

        #pprint(s)

        s = get_token_sentiment(tokens)
        df_s = pd.DataFrame([s], index=[i])
        df = df.append(df_s)

    df['pos'] = df.pos.apply(sum)
    df['neg'] = df.neg.apply(sum)
    print(df)


def get_token_sentiment(tokens, max_polarity_only=True):
    sentiments = {}

    for token in tokens:
        attrs = token.split('_')
        if len(attrs) != 3:
            continue
        #print(attrs)
        word = attrs[0]
        lemma = attrs[1]
        pos = attrs[2]
        if len(pos) > 0:
            pos = pos.lower()[0]
            if pos == 'j':
                pos = 'a'
        #print(pos)
        if pos not in ['a', 'n', 'r', 'v']:
            continue
        query = lemma + '.' + pos + '.01' # first (most common) sense
        #print(query)
        try:
            swn_syn = swn.senti_synset(query)
        except WordNetError as e:
            #print('-- blah.', word, lemma, pos)
            continue
        #print(swn_syn)
        pos = swn_syn.pos_score()
        neg = swn_syn.neg_score()

        # ignore entriees wiih  no  polarity scores (neutral)
        if pos == 0 and neg == 0:
            continue
        
        # if we only want the  max and values are equal, randomly  choose one
        if pos == neg and max_polarity_only:
            p = random.choice([0, 1])
            
            if p == 0:
                tmp = sentiments.get('pos_words', [])
                tmp.append(word)
                sentiments['pos_words'] = tmp
                tmp = sentiments.get('pos', [])
                tmp.append(pos)
                sentiments['pos'] = tmp
            
            elif p == 1:
                tmp = sentiments.get('neg_words', [])
                tmp.append(word)
                sentiments['neg_words'] = tmp
                tmp = sentiments.get('neg', [])
                tmp.append(-neg)
                sentiments['neg'] = tmp
            
            continue

        # equal, take  both
        if pos == neg:
            tmp = sentiments.get('pos_words', [])
            tmp.append(word)
            sentiments['pos_words'] = tmp
            tmp = sentiments.get('pos', [])
            tmp.append(pos)
            sentiments['pos'] = tmp

            tmp = sentiments.get('neg_words', [])
            tmp.append(word)
            sentiments['neg_words'] = tmp
            tmp = sentiments.get('neg', [])
            tmp.append(-neg)
            sentiments['neg'] = tmp
            
            continue

        # otherwise, avoid  both zeros
        if pos > 0 and pos > neg:
            tmp = sentiments.get('pos_words', [])
            tmp.append(word)
            sentiments['pos_words'] = tmp
            tmp = sentiments.get('pos', [])
            tmp.append(pos)
            sentiments['pos'] = tmp
            if max_polarity_only:
                continue
        
        if neg > 0:
            tmp = sentiments.get('neg_words', [])
            tmp.append(word)
            sentiments['neg_words'] = tmp
            tmp = sentiments.get('neg', [])
            tmp.append(-neg)
            sentiments['neg'] = tmp

    sentiments['pos'] = sum(sentiments.get('pos', [0]))
    sentiments['neg'] = sum(sentiments.get('neg', [0]))
    
    return sentiments


def analyse_tokens(tokens_list, i):
    df = pd.DataFrame(columns=['neg', 'neg_words', 'pos', 'pos_words'])
    
    t = len(tokens_list)
    
    for tokens in tokens_list:
        #print(tokens)
        #print('=====')
        s = get_token_sentiment(tokens)
        #print(s)
        df_s = pd.DataFrame([s], index=[0])
        df = df.append(df_s)
        if i % 100 == 0:
            print(i, '/', t)
        i += 1

    #print(df)
    #df['pos'].fillna(value=0, inplace=True)
    #df['neg'].fillna(value=0, inplace=True)
    #df['pos'] = df.pos.apply(sum)
    #df['neg'] = df.neg.apply(sum)
    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)

    # fill empty values
    for row in df.loc[df.pos_words.isnull(), 'pos_words'].index:
        df.at[row, 'pos_words'] = []

    for row in df.loc[df.neg_words.isnull(), 'neg_words'].index:
        df.at[row, 'neg_words'] = []
    
    return df


def format_results(sentiments):
    """
    Format to save offsets in individual files
    CHECK...format of output from get_token_sentiment may have changed
    """
    df = pd.DataFrame(columns={'pos', 'neg', 'pos_words', 'neg_words'})
    
    d = [[i, *v] for i, v in [(k, y) for k, v in sentiments.items() for y in v]]
    pd.DataFrame(d, columns=['cat', 'word', 'start', 'end', 'score'])


def process(ctype):
    print('-- Loading data for ' + ctype + '...', end='')
    df = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/' + ctype + '_30_text_pp_lemma.pickle')
    #df = pd.read_pickle('df_swn_test.pickle')
    print('Done.')
    
    print('-- Analysing text with SentiWordeNet...', end='')
    #df_sent = df['tokens_' + ctype].apply(analyse_tokens)
    t0 = time()
    i = 0
    df_sent = analyse_tokens(df['tokens_' + ctype], i)
    t1 = time()
    print('Done.')
    df = pd.concat([df, df_sent], axis=1)
    df.drop('tokens_' + ctype, axis=1, inplace=True)
    
    pout = BASE_DIR_Z + 'data/' + ctype + '_30_text_pp_swn_excl.pickle'
    df.to_pickle(pout)
    print('-- Wrote file:', pout)
    t2 = time()
    
    print('time 1:', (t1 - t0))
    print('time 2:', (t2 - t1))


if __name__ == '__main__':
    from time import time
    process('case')
    process('control')
