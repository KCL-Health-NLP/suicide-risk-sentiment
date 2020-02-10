# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:55:43 2018

@author: ABittar

This script contains utility functions to prepare data for experimaents on sentiment in clinical
notes of a data set used for suicide risk assessment.
"""

import numpy as np
import os
import pandas as pd
import pickle
import re
import spacy

from collections import Counter
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Data and resource directories mapped to the T and Z drives
BASE_DIR_T = 'T:/Andre Bittar/workspace/ehost-it/suicide-sentiment/'
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'
RESOURCE_DIR = BASE_DIR_T + 'resources/'

ptype = 'control'

print('-- Loading spaCy model for English...', end='')
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
print('Done.')


def plot(df, emotion):
    df[['Date_ord_norm', emotion]].groupby('Date_ord_norm').sum().reset_index().plot(kind='bar', x='Date_ord_norm', y=emotion)


def load_lexicon(path):
    """
    Load a lexicon from a text file - file must be in EmoLex format
    (word<tab>emotion<tab>value):
    ...
    abandon	sadness	1
    abandon	surprise	0
    abandon	trust	0
    abandoned	anger	1
    abandoned	anticipation	0
    ...

    For EmoLex use: 'data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    For LIWC lexicons use: data/LIWC_2007_pos_neg_emo.txt
    For Opinion use: data/opinion-lexicon-English/positive-words.txt
                     data/opinion-lexicon-English/negative-words.txt

    """
    
    print('-- Preparing lexicon...', end='')
    entries = [line.split('\t') for line in open(path, 'r').read().split('\n')
               if line != '' and '\t0' not in line]
    
    df_lex = pd.DataFrame(entries, columns=['word', 'emotion', 'value'])
    df_lex['value'] = df_lex.value.astype(int)
    print('Done.')
    
    return df_lex


def load_data(ptype):
    """
    Just load the data from the DataFrame.
    """
    print('-- Loading data...', end='')
    df = pd.read_pickle(BASE_DIR_Z + 'data/' + ptype + '_30_text_ordinal_dates.pickle')
    print('Done.')

    return df


def get_offsets_by_token(df_lex, text, match_on):
    """
    Get the character offsets of the emotion tokens in the text.
    Match on either spaCy word form or lemma.
    """
    doc = nlp(text)
    emotion_offsets = {}

    for token in doc:
        if token.lemma_ not in stopwords.words('english') and \
        not token.is_punct and not token.is_space:
            if match_on == 'lemma':
                emos = df_lex.loc[df_lex.word == token.lemma_].emotion
            elif match_on == 'word':
                emos = df_lex.loc[df_lex.word == token.text].emotion
            else:
                raise Exception('-- Error: invalid match_on ' + match_on + ' choose "word" or "lemma".')
            for e in emos:
                tmp = emotion_offsets.get(e, [])
                start = token.idx
                end = start + len(token.text)
                tmp.append((token.text, start, end))
                emotion_offsets[e] = tmp
    
    return emotion_offsets


def get_offsets_by_regex(df_lex, text, escape=False):
    """
    Get the character offsets of the emotion tokens in the text.
    Match on regex.
    This is really inefficient for anything but the smallest of lexicons
    due to the cost of regex search.
    """
    #c = df_lex.word.apply(lambda x: [[match[0], match.start(), match.start() + len(match[0])] for match in re.finditer(x, text, flags=re.I)])
    emotion_offsets = {}
    for i, row in df_lex.iterrows():
        if escape:
            regex = r'\b' + re.escape(row.word) + r'\b'
        else:
            regex = r'\b' + row.word + r'\b'
        print(regex)
        matches = re.finditer(regex, text, flags=re.I + re.U)

        if matches is not None:
            for match in matches:
                start = match.start()
                end = start + len(match[0])
                tmp = emotion_offsets.get(row.emotion, [])
                tmp.append((match[0], start, end))
                emotion_offsets[row.emotion] = tmp
    
    return emotion_offsets


def batch_get_offsets(lexicon_path, lexicon_name, method, token_match_on=None, case=True, control=True):
    """
    Get offsets of all words from the lexicon (EmoLex format) that are found in
    the text, for cases and controls. Stores the offsets and word lemma in a 
    dictionary that is saved as a pickle file.
    method='token' or 'regex'
    Run this before load_extracted_words_to_dataframe()
    """
    
    if not (case or control):
        print('-- batch_get_offsets: doing nothing (case and control values are False)')
        return
    
    if method not in ['token', 'regex']:
        raise Exception('-- Error: invalid method ' + method + ' choose "token" or "regex".')
    
    if method == 'token' and token_match_on not in ['word', 'lemma']:
        raise Exception('-- Error: invalid token match_on ' + str(token_match_on) + ' choose "word" or "lemma".')
    
    df_lex = load_lexicon(lexicon_path)
    
    n = 0
    m = 0
    
    if method == 'regex':
        base_dir = BASE_DIR_Z + 'data/emotion/' + lexicon_name + '_' + method + '_offsets'
    else:
        base_dir = BASE_DIR_Z + 'data/emotion/' + lexicon_name + '_' + method + '_' + token_match_on + '_offsets'
    
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    
    if case:
        print('-- Processing cases with ' + lexicon_name + '...', end='')
        if not os.path.isdir(base_dir + '/case'):
            os.makedirs(base_dir + '/case')
        df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates_p2.pickle')
        
        j = 0
        l = len(df_case)
        for group in df_case.groupby(by=['pk', 'CN_Doc_ID']):
            for i, row in group[1].iterrows():
                if method == 'token':
                    e = get_offsets_by_token(df_lex, row.text_case, match_on=token_match_on)
                elif method == 'regex':
                    e = get_offsets_by_regex(df_lex, row.text_case)
                fname = 'case_' + row.pk + '_' + row.CN_Doc_ID + '_' + str(i) + '.pickle'
                if len(e) > 0:
                    pickle.dump(e, open(base_dir + '/case/' + fname, 'wb'))
                n += 1
            if j % 1000 == 0:
                print(str(j) + '/' + str(l))
            j += 1
        print('Done.')
    
    if control:
        print('-- Processing controls with ' + lexicon_name + '...', end='')
        if not os.path.isdir(base_dir + '/control'):
            os.makedirs(base_dir + '/control')
        df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates_p2.pickle')

        j = 0
        l = len(df_control)        
        for group in df_control.groupby(by=['pk', 'CN_Doc_ID']):
            for i, row in group[1].iterrows():
                if method == 'token':
                    e = get_offsets_by_token(df_lex, row.text_control, match_on=token_match_on)
                elif method == 'regex':
                    e = get_offsets_by_regex(df_lex, row.text_control)                
                fname = 'control_' + row.pk + '_' + row.CN_Doc_ID + '_' + str(i) + '.pickle'
                if len(e) > 0:
                    pickle.dump(e, open(base_dir + '/control/' + fname, 'wb'))
                m += 1
            if j % 1000 == 0:
                print(str(j) + '/' + str(l))
            j += 1
        print('Done.')
    
    print('-- Processed files:', n, 'cases,', m, 'controls')


def show_words(ptype, df=None):
    """
    Output the words in the text that were matched with the dictionary (EmoLex).
    """
    edir = None
    label = None
    
    if ptype == 'case':
        if df is None:
            df = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates_p2.pickle')
        edir = BASE_DIR_Z + 'data/emotion/emotion_offsets/case'
        label = 'text_case'
    elif ptype == 'control':
        if df is None:
            df = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates_p2.pickle')
        edir = BASE_DIR_Z + 'data/emotion/emotion_offsets/control'
        label = 'text_control'
    else:
        print('-- Invalid ptype argument:', ptype)
        return

    print('-- Getting', ptype, 'file list...', end='')
    files = [f for f in os.listdir(edir) if f.startswith(ptype)]
    print('Done.')
    
    words = []
    n = 0
    
    for f in files:
        match = re.search('_([^_]+)_([^_]+)_([0-9]+)\.pickle', f)
        #pk = None
        #cndocid = None
        index = None

        if match is not None:
            #pk = match.group(1)
            #cndocid = match.group(2)
            index = int(match.group(3))
        else:
            #print(match)
            print('-- Invalid filename (ignoring):', f)
            continue

        # don't go past the last index
        if index > df.last_valid_index():
            print('-- File out of bounds (ignoring):', f)
            break
        
        fin = os.path.join(edir, f)
        try:
            emo_set = pickle.load(open(fin, 'rb'))
        except Exception as e:
            print('fin=', fin)
            print(e)
            return words, df
        
        text = df.iloc[index][label]
        
        for emo_key in emo_set:
            for emo in emo_set[emo_key]:
                lemma = emo[0] # change others to 1 and 2 to fit new structure
                start = emo[1]
                end = emo[2]
                word = text[start:end]
                words.append((word, lemma, emo_key))
            #print(f, emo_key, word, start, end)
        
        n += 1
        if n % 1000 == 0:
            print(n)
    
    pout = edir + '/matched_emotion_words.pickle'
    pickle.dump(words, open(pout, 'wb'))

    print('-- Wrote file:', pout)
    
    return words, df


def filter_sentiwordnet(use_mwe=True):
    """
    Filter out unwanted material from SentiWordNet.
    Expand all rows that contain multiple senses so there is only 1 sense per row.
    Remove senses that have 0 score for positive and negative.
    use_mwe: optionally remove senses that are multiword expressions (appear with _ as separator)
    Store synset terms as words without the #number
    """
    df_swn = pd.read_csv(RESOURCE_DIR + 'SentiWordNet_3.0.0_20130122.txt', sep='\t', header=26)
    df_swn.rename(columns={'# POS': 'POS'}, inplace=True)
    df_swn['ID'] = df_swn.ID.astype(str)
    df_swn['POS_ID'] = df_swn['POS'] + df_swn['ID']
    
    b = pd.DataFrame(df_swn.SynsetTerms.str.split(' ').tolist(), index=df_swn['POS_ID'])
    b = b.stack(dropna=True)
    b = b.reset_index()[[0, 'POS_ID']] # var1 variable is currently labeled 0
    b.columns = ['SynsetTerms', 'POS_ID'] # renaming var1
    
    df_swn.drop('SynsetTerms', axis=1, inplace=True)
    df_swn = df_swn.merge(b, on='POS_ID')
    df_swn.drop(df_swn.last_valid_index(), inplace=True)
    
    if use_mwe == False:
        df_swn = df_swn.loc[~((df_swn.NegScore == 0) & (df_swn.PosScore == 0)) & ~df_swn.SynsetTerms.str.contains('_')]
    else:
        df_swn = df_swn.loc[~((df_swn.NegScore == 0) & (df_swn.PosScore == 0))]
    df_swn['word'] = df_swn.SynsetTerms.apply(lambda x: re.sub('#[0-9]+', '', x))
    #df_swn = df_swn.loc[~((df_swn.PosScore == 0) & (df_swn.NegScore == 0))] # remove any entries that have no polarity score
    
    return df_swn


def count_sentiments(ctype, df_swn):
    """
    Count the number of occurrences of each (pos/neg) sentiment word in the 
    SentiWordNet lexicon found in the tokenised text.
    Calculate sum of positive and negative scores for each document.
    """
    df = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/case_30_text_pp_lemma.pickle')
    df['tokens_case'] = df.tokens_case.apply(lambda x: [y.split('_')[0] for y in x])
    df.tokens_case.fillna(value='', inplace=True)
    df['swn_pos'] = 0
    df['swn_neg'] = 0
    df['swn_words'] = np.empty((len(df.index), 0)).tolist() # initialise with empty lists

    # this is really inefficient
    for index, row in df.iterrows():
        for token in row['tokens_' + ctype]:
            sents = df_swn.loc[df_swn.word == token]
            if len(sents.index) > 0:
                df.iloc[index, df.columns.get_loc('swn_pos')] += sents.PosScore.sum()
                df.iloc[index, df.columns.get_loc('swn_neg')] += sents.NegScore.sum()
                tmp = df.iat[index, df.columns.get_loc('swn_words')]
                tmp.extend(sents.word.tolist())
                df.iat[index, df.columns.get_loc('swn_words')] = tmp
        if index % 100 == 0:
            print(index)
                
    return df


def count_emotions(ptype, df_lex):
    """
    Count the number of occurrences of each emotion in the lexicon found in the
    tokenised text.
    """
    df = load_data(ptype)

    print('-- Preprocessing text...', end='')
    df['tokens_' + ptype] = df['text_' + ptype].apply(lambda x: [token.lemma_ for token in nlp(x) if token.lemma_ not in stopwords.words('english')
                                          and not token.is_punct
                                          and not token.is_space])
    print('Done.')

    for e in df_lex.emotion.unique():
        df[e] = 0

    # this is really inefficient
    for index, row in df.iterrows():
        print(index, '/', len(df))
        for token in row['tokens_' + ptype]:
            emos = df_lex.loc[df_lex.word == token].emotion
            for e in emos:
                df.iloc[index, df.columns.get_loc(e)] += 1

    df['len'] = df['text_' + ptype].apply(lambda x: len(x.split()))

    df = df.groupby('Date_ord_norm').sum().reset_index()

    for e in df_lex.emotion.unique():
        df[e + '_freq'] = df[e] / df.len

    return df


def plot_emotions(df_case=None, df_control=None):
    """
    Plot frequency of emotions from EmoLex matched within the text for each
    day from 0-30 days prior to case admissions.
    """
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/emotion/case_emotion_freq.pickle')
    df_control = pd.read_pickle(BASE_DIR_Z + 'data/emotion/control_emotion_freq.pickle')
    
    emos = sorted(['trust', 'fear', 'negative', 'sadness', 'anger', 'surprise', 'positive', 'disgust', 'joy', 'anticipation'])
    
    tmp_case = df_case[[e + '_freq' for e in emos]].rename(columns=dict(zip([e + '_freq' for e in emos], [e + '_case' for e in emos])))
    tmp_control = df_control[[e + '_freq' for e in emos]].rename(columns=dict(zip([e + '_freq' for e in emos], [e + '_control' for e in emos])))
    tmp = pd.concat([tmp_case, tmp_control], axis=1)
    tmp.index += 1
    tmp = tmp * 100
    
    for e in emos:
        plot_path = BASE_DIR_T + 'plots/cc_30d_'  + '_emotion_' + e + '_area.png'
        
        tmp.rename(columns={e + '_case': 'case ' + e, e + '_control': 'control ' + e}, inplace=True)
        ax = tmp[['case ' + e, 'control ' + e]].plot(kind='area', stacked=False)
        ax.set_xlabel('Days prior to (case) admission date', fontsize=20)
        ax.set_ylabel('Emotion frequency (per 100 words)', fontsize=20)
        ax.set_ylim([0, 3])
        ax.set_title('Frequency of ' + e + ' words', fontsize=20)
        ax.tick_params(labelsize=20)
        
        ax.collections[0].set_color('#D9D75E')
        ax.collections[1].set_color('#76A0BD')
        ax.get_lines()[0].set_color('#D9D75E')
        ax.get_lines()[1].set_color('#76A0BD')
        ax.collections[0].set_linewidth(2)
        ax.collections[1].set_linewidth(2)
        ax.legend(fontsize=20)
        
        fig = ax.get_figure()
        fig.set_size_inches(10, 7)
        fig.savefig(plot_path, bbox_inches='tight')


def get_same_number_of_tokens():
    """
    Produce two DataFrames with the same number of tokens.
    This is needed to ensure we are looking at the same amount of text for cases
    and controls when applying Pattern and EmoLex or other tools.
    Returns a control data frame that has approximately the same number of tokens
    as all the cases.
    """
    
    print('-- Loading data...', end='')
    
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates.pickle')
    df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates.pickle')
    
    print('Done.')

    print('-- Counting tokens..', end='')

    df_case['len'] = df_case.text_case.apply(lambda x: len(x.split()))
    df_control['len'] = df_control.text_control.apply(lambda x: len(x.split()))
    
    print('Done.')
    
    case_tokens = df_case.len.sum()
    
    ct = 0
    idx = []
    for i, row in df_control.iterrows():
        ct += row.len
        if ct >= case_tokens:
            break
        idx.append(i)

    df_control = df_control.loc[idx]
    
    print('Number of tokens for cases   :', case_tokens)
    print('Number of tokens for controls:', df_control.len.sum())
    
    return df_control


def plot_words_per_day(emo_source):
    """
    Plot the mean number of words per day for cases and controls.
    """
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/case_30_text_per_day_' + emo_source + '.pickle')
    df_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/control_30_text_per_day_' + emo_source + '.pickle')

    df_case['len_case'] = df_case.text_case.apply(lambda x: len(x.split()))
    df_control['len_control'] = df_control.text_control.apply(lambda x: len(x.split()))

    df_case = df_case.groupby('day').mean().reset_index()[['day', 'len_case']]
    df_control = df_control.groupby('day').mean().reset_index()[['day', 'len_control']]
    
    xlabel = 'Days prior to (case) admission date'
    ylabel = 'Average number of words (raw counts)'

    df_case['len_control'] = df_control['len_control']
    ax = df_case.plot(kind='line', x='day', stacked=False, title='Words for ' + emo_source)
    ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig = ax.get_figure()
    
    return df_case


def plot_emotions_per_day(emo_source, wpm=False):
    """
    Plot the output of sentiment analysis per day for cases and controls.
    """
    
    print('-- Plotting sentiment for', emo_source)
    
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/case_30_text_per_day_' + emo_source + '.pickle')
    df_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/control_30_text_per_day_' + emo_source + '.pickle')
    
    df_case.rename(columns={'pos_' + emo_source: 'pos_case', 'neg_' + emo_source: 'neg_case'}, inplace=True)
    df_control.rename(columns={'pos_' + emo_source: 'pos_control', 'neg_' + emo_source: 'neg_control'}, inplace=True)
    
    df_case_pos = df_case.groupby('day').mean().reset_index()[['day', 'pos_case']]
    df_control_pos = df_control.groupby('day').mean().reset_index()[['day', 'pos_control']]

    df_case_neg = df_case.groupby('day').mean().reset_index()[['day', 'neg_case']]
    df_control_neg = df_control.groupby('day').mean().reset_index()[['day', 'neg_control']]
    
    xlabel = 'Days prior to (case) admission date'
    ylabel = 'Average valence score'

    # normalise as words per million (wpm)
    if wpm:
        df_case['len'] = df_case.text_case.apply(lambda x: len(x.split()))
        df_control['len'] = df_control.text_control.apply(lambda x: len(x.split()))

        df_case['pos_case'] = df_case['pos_case'] / df_case['len'] * 1000000
        df_control['pos_control'] = df_control['pos_control'] / df_control['len'] * 1000000

        df_case['neg_case'] = df_case['neg_case'] / df_case['len'] * 1000000
        df_control['neg_control'] = df_control['neg_control'] / df_control['len'] * 1000000
        
        ylabel = 'Frequency (words per million)'

    df_case_pos['pos_control'] = df_control_pos['pos_control']
    ax = df_case_pos.plot(kind='line', x='day', stacked=False, title='Positive sentiment for ' + emo_source)
    ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig = ax.get_figure()
    fig.savefig(BASE_DIR_Z + 'data/clpsych/plots/cc_' + emo_source + '_pos_scores.png', bbox_inches='tight')

    df_case_neg['neg_control'] = df_control_neg['neg_control']
    ax = df_case_neg.plot(kind='line', x='day', stacked=False, title='Negative sentiment for ' + emo_source)
    ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig = ax.get_figure()
    fig.savefig(BASE_DIR_Z + 'data/clpsych/plots/cc_' + emo_source + '_neg_scores.png', bbox_inches='tight')

    return df_case, df_control


def plot_emotions_per_day_proportional(emo_source):
    """
    Plot the proportional output of sentiment analysis for cases and controls per day
    """
    
    print('-- Plotting sentiment for', emo_source)
    
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/case_30_text_per_day_' + emo_source + '.pickle')
    df_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/control_30_text_per_day_' + emo_source + '.pickle')
    
    df_case.rename(columns={'pos_' + emo_source: 'pos_case', 'neg_' + emo_source: 'neg_case'}, inplace=True)
    df_control.rename(columns={'pos_' + emo_source: 'pos_control', 'neg_' + emo_source: 'neg_control'}, inplace=True)
    
    df_case_pos = df_case.groupby('day').mean().reset_index()[['day', 'pos_case']]
    df_control_pos = df_control.groupby('day').mean().reset_index()[['day', 'pos_control']]

    df_case_neg = df_case.groupby('day').mean().reset_index()[['day', 'neg_case']]
    df_control_neg = df_control.groupby('day').mean().reset_index()[['day', 'neg_control']]
    
    df_case_sent = df_case_pos.merge(df_case_neg)
    df_case_sent['sent_total'] = df_case_sent.pos_case + df_case_sent.neg_case
    df_case_sent['pos_prop_case'] = df_case_sent.pos_case / df_case_sent.sent_total
    df_case_sent['neg_prop_case'] = df_case_sent.neg_case / df_case_sent.sent_total

    df_control_sent = df_control_pos.merge(df_control_neg)
    df_control_sent['sent_total'] = df_control_sent.pos_control + df_control_sent.neg_control
    df_control_sent['pos_prop_control'] = df_control_sent.pos_control / df_control_sent.sent_total
    df_control_sent['neg_prop_control'] = df_control_sent.neg_control / df_control_sent.sent_total
    
    xlabel = 'Days prior to (case) admission date'
    ylabel = 'Average propotion of positive sentiment'
    
    df_sent_prop_pos = pd.concat([df_case_sent.day, df_case_sent.pos_prop_case, df_control_sent.pos_prop_control], axis=1)
    df_sent_prop_neg = pd.concat([df_case_sent.day, df_case_sent.neg_prop_case, df_control_sent.neg_prop_control], axis=1)
    
    ax = df_sent_prop_pos.plot(kind='line', x='day', stacked=False, title='Proportion of positive sentiment for ' + emo_source)
    ax.invert_xaxis()
    #ax.set_ylim([0, 2.5]) # TODO get values from data and normalise wrt to other emo_sources
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(['case', 'control'])
    fig = ax.get_figure()
    fig.savefig(BASE_DIR_Z + 'data/clpsych/plots/cc_pos_scores_prop_' + emo_source + '.png', bbox_inches='tight')

    if False:
        ylabel = 'Average propotion of negative sentiment'

        ax = df_sent_prop_neg.plot(kind='line', x='day', stacked=False, title='Proportion of negative sentiment for ' + emo_source)
        ax.invert_xaxis()
        #ax.set_ylim([0, 2.5]) # TODO get values from data and normalise wrt to other emo_sources
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(['case', 'control'])
        fig = ax.get_figure()
    
    return df_sent_prop_pos


def plot_emotions_per_day_proportional_all(emo_source):
    """
    Plot the proportional output of sentiment analysis for cases and controls per day
    """
    
    print('-- Plotting sentiment for', emo_source)
    
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/case_30_text_per_day_' + emo_source + '.pickle')
    df_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/control_30_text_per_day_' + emo_source + '.pickle')
        
    df_case_pos = df_case.groupby('day').mean().reset_index()[['day', 'pos_case_'  + emo_source]]
    df_control_pos = df_control.groupby('day').mean().reset_index()[['day', 'pos_control_' + emo_source]]

    df_case_neg = df_case.groupby('day').mean().reset_index()[['day', 'neg_case_' + emo_source]]
    df_control_neg = df_control.groupby('day').mean().reset_index()[['day', 'neg_control_'  + emo_source]]
    
    df_case_sent = df_case_pos.merge(df_case_neg)
    df_case_sent['sent_total'] = df_case_sent['pos_case_' + emo_source] + df_case_sent['neg_case_' + emo_source]
    df_case_sent['pos_prop_case_' + emo_source] = df_case_sent['pos_case_' + emo_source] / df_case_sent.sent_total
    df_case_sent['neg_prop_case_' + emo_source] = df_case_sent['neg_case_' + emo_source] / df_case_sent.sent_total

    df_control_sent = df_control_pos.merge(df_control_neg)
    df_control_sent['sent_total'] = df_control_sent.pos_control + df_control_sent.neg_control
    df_control_sent['pos_prop_control_' + emo_source] = df_control_sent['pos_control_' + emo_source] / df_control_sent.sent_total
    df_control_sent['neg_prop_control_' + emo_source] = df_control_sent['neeg_control_' + emo_source] / df_control_sent.sent_total

    df_sent_prop_pos = pd.concat([df_case_sent.day, df_case_sent['pos_prop_case_' + emo_source], df_control_sent['pos_prop_control_' + emo_source]], axis=1)
    df_sent_prop_neg = pd.concat([df_case_sent.day, df_case_sent['neg_prop_case_' + emo_source], df_control_sent['neg_prop_control_' + emo_source]], axis=1)

    return df_sent_prop_pos, df_sent_prop_neg


def plot_emotions_per_day_proportional_emolex():
    """
    Plot the proportional output of sentiment analysis for cases and controls per day
    """
    
    print('-- Plotting sentiment for emolex')
    
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/case_30_text_per_day_emolex.pickle')
    df_control = pd.read_pickle(BASE_DIR_Z + 'data/clpsych/text_per_day/control_30_text_per_day_emolex.pickle')
    
    for emo in ['ang', 'ant', 'dis', 'fea', 'joy', 'sad', 'sur', 'tru']:
        df_case.rename(columns={emo: emo + '_case'}, inplace=True)
        df_control.rename(columns={emo: emo + '_control'}, inplace=True)
    
    df_case_ang = df_case.groupby('day').mean().reset_index()[['day', 'ang_case']]
    df_control_ang = df_control.groupby('day').mean().reset_index()[['day', 'ang_control']]

    df_case_ant = df_case.groupby('day').mean().reset_index()[['day', 'ant_case']]
    df_control_ant = df_control.groupby('day').mean().reset_index()[['day', 'ant_control']]

    df_case_dis = df_case.groupby('day').mean().reset_index()[['day', 'dis_case']]
    df_control_dis = df_control.groupby('day').mean().reset_index()[['day', 'dis_control']]

    df_case_fea = df_case.groupby('day').mean().reset_index()[['day', 'fea_case']]
    df_control_fea = df_control.groupby('day').mean().reset_index()[['day', 'fea_control']]

    df_case_joy = df_case.groupby('day').mean().reset_index()[['day', 'joy_case']]
    df_control_joy = df_control.groupby('day').mean().reset_index()[['day', 'joy_control']]

    df_case_sad = df_case.groupby('day').mean().reset_index()[['day', 'sad_case']]
    df_control_sad = df_control.groupby('day').mean().reset_index()[['day', 'sad_control']]

    df_case_sur = df_case.groupby('day').mean().reset_index()[['day', 'sur_case']]
    df_control_sur = df_control.groupby('day').mean().reset_index()[['day', 'sur_control']]

    df_case_tru = df_case.groupby('day').mean().reset_index()[['day', 'tru_case']]
    df_control_tru = df_control.groupby('day').mean().reset_index()[['day', 'tru_control']]
    
    # TODO these emotions are supposed to be binary opposites - so we shouldn't plot them all together
    
    df_case_sent = df_case_ang.merge(df_case_ant).merge(df_case_dis).merge(df_case_fea).merge(df_case_joy).merge(df_case_sad).merge(df_case_sur).merge(df_case_tru)
    df_case_sent['sent_total'] = df_case_sent['ang_case'] + df_case_sent['ant_case'] + df_case_sent['dis_case'] + df_case_sent['fea_case'] + df_case_sent['joy_case'] + df_case_sent['sad_case'] + df_case_sent['sur_case'] + df_case_sent['tru_case']
    df_case_sent['ang_prop_case'] = df_case_sent['ang_case'] / df_case_sent.sent_total
    df_case_sent['ant_prop_case'] = df_case_sent['ant_case'] / df_case_sent.sent_total
    df_case_sent['dis_prop_case'] = df_case_sent['dis_case'] / df_case_sent.sent_total
    df_case_sent['fea_prop_case'] = df_case_sent['fea_case'] / df_case_sent.sent_total
    df_case_sent['joy_prop_case'] = df_case_sent['joy_case'] / df_case_sent.sent_total
    df_case_sent['sad_prop_case'] = df_case_sent['sad_case'] / df_case_sent.sent_total
    df_case_sent['sur_prop_case'] = df_case_sent['sur_case'] / df_case_sent.sent_total
    df_case_sent['tru_prop_case'] = df_case_sent['tru_case'] / df_case_sent.sent_total

    df_control_sent = df_control_ang.merge(df_control_ant).merge(df_control_dis).merge(df_control_fea).merge(df_control_joy).merge(df_control_sad).merge(df_control_sur).merge(df_control_tru)
    df_control_sent['sent_total'] = df_control_sent['ang_control'] + df_control_sent['ant_control'] + df_control_sent['dis_control'] + df_control_sent['fea_control'] + df_control_sent['joy_control'] + df_control_sent['sad_control'] + df_control_sent['sur_control'] + df_control_sent['tru_control']
    df_control_sent['ang_prop_control'] = df_control_sent['ang_control'] / df_control_sent.sent_total
    df_control_sent['ant_prop_control'] = df_control_sent['ant_control'] / df_control_sent.sent_total
    df_control_sent['dis_prop_control'] = df_control_sent['dis_control'] / df_control_sent.sent_total
    df_control_sent['fea_prop_control'] = df_control_sent['fea_control'] / df_control_sent.sent_total
    df_control_sent['joy_prop_control'] = df_control_sent['joy_control'] / df_control_sent.sent_total
    df_control_sent['sad_prop_control'] = df_control_sent['sad_control'] / df_control_sent.sent_total
    df_control_sent['sur_prop_control'] = df_control_sent['sur_control'] / df_control_sent.sent_total
    df_control_sent['tru_prop_control'] = df_control_sent['tru_control'] / df_control_sent.sent_total

    df_sent_prop_ang = pd.concat([df_case_sent.day, df_case_sent['ang_prop_case'], df_control_sent['ang_prop_control']], axis=1)
    df_sent_prop_ant = pd.concat([df_case_sent.day, df_case_sent['ant_prop_case'], df_control_sent['ant_prop_control']], axis=1)
    df_sent_prop_dis = pd.concat([df_case_sent.day, df_case_sent['dis_prop_case'], df_control_sent['dis_prop_control']], axis=1)
    df_sent_prop_fea = pd.concat([df_case_sent.day, df_case_sent['fea_prop_case'], df_control_sent['fea_prop_control']], axis=1)
    df_sent_prop_joy = pd.concat([df_case_sent.day, df_case_sent['joy_prop_case'], df_control_sent['joy_prop_control']], axis=1)
    df_sent_prop_sad = pd.concat([df_case_sent.day, df_case_sent['sad_prop_case'], df_control_sent['sad_prop_control']], axis=1)
    df_sent_prop_sur = pd.concat([df_case_sent.day, df_case_sent['sur_prop_case'], df_control_sent['sur_prop_control']], axis=1)
    df_sent_prop_tru = pd.concat([df_case_sent.day, df_case_sent['tru_prop_case'], df_control_sent['tru_prop_control']], axis=1)

    emo_map = {
            'anger': df_sent_prop_ang, 
            'anticipaction': df_sent_prop_ant,
            'disgust': df_sent_prop_dis,
            'fear': df_sent_prop_fea,
            'joy': df_sent_prop_joy,
            'sadness': df_sent_prop_sad,
            'surprise': df_sent_prop_sur,
            'trust': df_sent_prop_tru
            }

    for emo in emo_map:
        xlabel = 'Days prior to (case) admission date'
        ylabel = 'Average propotion of ' + emo
        
        df = emo_map[emo]
        ax = df.plot(kind='line', x='day', stacked=False, title='Proportion of ' + emo)
        ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(['case', 'control'])
        fig = ax.get_figure()
        fig.savefig(BASE_DIR_Z + 'data/clpsych/plots/cc_' + emo + '_scores_prop_emolex.png', bbox_inches='tight')

    return df_sent_prop_ang, df_sent_prop_ant, df_sent_prop_dis, df_sent_prop_fea, df_sent_prop_joy, df_sent_prop_sad, df_sent_prop_sur, df_sent_prop_tru


def plot_pattern_output(emo_source, use_equal_token_num=False):
    """
    OLD method that still works, adapted for other sources...
    Plot the output of Pattern sentiment analysis for cases and controls
    """
    df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates_' + emo_source + '_p2.pickle')
    
    if use_equal_token_num:
        df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates_' + emo_source + '_p2_equal_num_tokens.pickle')
    else:
        df_control = pd.read_pickle(BASE_DIR_Z + 'data/control_30_text_ordinal_dates_' + emo_source + '_p2.pickle')
    
    df_case.rename(columns={'pos': 'pos_case', 'neg': 'neg_case'}, inplace=True)
    df_control.rename(columns={'pos': 'pos_control', 'neg': 'neg_control'}, inplace=True)
    
    df_case_pos = df_case.groupby('Date_ord_norm').mean().reset_index()[['Date_ord_norm', 'pos_case']]
    df_control_pos = df_control.groupby('Date_ord_norm').mean().reset_index()[['Date_ord_norm', 'pos_control']]

    df_case_neg = df_case.groupby('Date_ord_norm').mean().reset_index()[['Date_ord_norm', 'neg_case']]
    df_control_neg = df_control.groupby('Date_ord_norm').mean().reset_index()[['Date_ord_norm', 'neg_control']]

    """ax = df_case_pos.plot(kind='area', x='Date_ord_norm')
    fig = ax.get_figure()
    fig.savefig(BASE_DIR_Z + 'data/emotion/pattern_output/case_pos_scores.png', bbox_inches='tight')
    
    ax = df_control_pos.plot(kind='area', x='Date_ord_norm')
    fig = ax.get_figure()
    fig.savefig(BASE_DIR_Z + 'data/emotion/pattern_output/control_pos_scores.png', bbox_inches='tight')

    ax = df_case_neg.plot(kind='area', x='Date_ord_norm')
    fig = ax.get_figure()
    fig.savefig(BASE_DIR_Z + 'data/emotion/pattern_output/case_neg_scores.png', bbox_inches='tight')
    
    ax = df_control_neg.plot(kind='area', x='Date_ord_norm')
    fig = ax.get_figure()
    fig.savefig(BASE_DIR_Z + 'data/emotion/pattern_output/control_neg_scores.png', bbox_inches='tight')
    """

    xlabel = 'Days prior to (case) admission date'
    ylabel = 'Average valence score'

    emo_source = emo_source.title()
    df_case_pos.rename(columns={'pos_case': 'case positive'}, inplace=True)
    df_case_pos['control positive'] = df_control_pos['pos_control']
    ax = df_case_pos.plot(kind='area', x='Date_ord_norm', title='Positive sentiment for ' + emo_source, stacked=False)
    y_max = max(df_case_pos['case positive'].max(), df_case_pos['control positive'].max())
    ax.set_ylim([0, 3])
    ax.set_title('Positive sentiment for ' + emo_source, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=20)
    fig = ax.get_figure()
    fig.set_size_inches(10, 7)
    fig.savefig(BASE_DIR_Z + 'data/emotion/pattern_output/cc_pos_scores_' + emo_source + '.png', bbox_inches='tight', dpi=300)

    df_case_neg.rename(columns={'neg_case': 'case negative'}, inplace=True)
    df_case_neg['control negative'] = df_control_neg['neg_control']
    ax = df_case_neg.plot(kind='area', x='Date_ord_norm', title='Negative sentiment for ' + emo_source, stacked=False)
    y_max = max(df_case_neg['case negative'].max(), df_case_neg['control negative'].max())
    ax.set_ylim([0, 3]) # 2.5
    ax.set_title('Negative sentiment for ' + emo_source, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=20)
    fig = ax.get_figure()
    fig.set_size_inches(10, 7)
    fig.savefig(BASE_DIR_Z + 'data/emotion/pattern_output/cc_neg_scores_' + emo_source + '.png', bbox_inches='tight', dpi=300)

    return df_case, df_control


def get_emotion_stats(ctype, emo_source, day='all', use_equal_token_num=False):
    """
    Output the list of emotion-term-counts sorted by number of occurrences.
    """
    
    pout = None
    if emo_source in ['afinn', 'emolex', 'opinion', 'pattern', 'swn']:
        if ctype == 'case':
            if day == 'all':
                d = pd.read_pickle(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/case_' + emo_source + '_cat_word_counts.pickle')
                pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/case_' + emo_source + '_cat_word_counts.txt'
            elif emo_source == 'pattern': # only for pattern for now
                d = pd.read_pickle(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/case_day_' + str(day) + '_word_counts.pickle')
                pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/case_' + emo_source + '_day_' + str(day) + '_cat_word_counts.txt'
            else:
                raise Exception('-- Per-day option not yet implemented for', emo_source)
        elif ctype == 'control':
            if use_equal_token_num:
                d = pd.read_pickle(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/' + emo_source + '_output/control_' + emo_source + '_cat_word_counts_equal_num_tokens.pickle')
                pout = BASE_DIR_Z + 'data/emotion/control_/' + emo_source + '_output/' + emo_source + '_output/' + emo_source + '_cat_word_counts_equal_num_tokens.txt'
            else:
                if day == 'all':
                    d = pd.read_pickle(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/control_' + emo_source + '_cat_word_counts.pickle')
                    pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/control_' + emo_source + '_cat_word_counts.txt'
                elif emo_source == 'pattern': # only for pattern for now
                    d = pd.read_pickle(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/control_day_' + str(day) + '_word_counts.pickle')
                    pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/control_' + emo_source + '_day_' + str(day) + '_cat_word_counts.txt'
                else:
                    raise Exception('-- Per-day option not yet implemented for', emo_source)
        else:
            raise Exception('-- Error: invalid ctype ' + ctype + ' choose "case" or "control".')
    else:
        raise Exception('-- Error: invalid emo_source ' + emo_source + ' choose "emolex" or "pattern".')

    # number of tokens is calculated over entire corpus by whitesapce split
    if ctype == 'case':
        num_tokens = 20187666
    elif ctype == 'control':
        num_tokens = 41909800

    with open(pout, 'w', encoding='utf-8') as fout:
        print('emotion\tword\tcount_' + ctype + '\tabs_freq_' + ctype + '\trel_freq_' + ctype,  file=fout)
        for cat in sorted(d.keys()):
            counts = d[cat]
            total = float(sum(counts.values()))
            for f in counts.most_common():
                word = f[0]
                count = f[1]
                abs_freq = f[1] / num_tokens
                rel_freq = f[1] / total
                print(cat + '\t' + word + '\t' + str(count) + '\t' + str(abs_freq) + '\t' + str(rel_freq),  file=fout)

    print('-- Wrote file:', pout)


def calculate_log_likelihood(df_cc):
    a = df_cc.count_case
    b = df_cc.count_control
    c = 20187666 # total words in corpus 1 (case)
    d = 41909800 # total words in corpus 2 (control)
    E1 = c * (a+b) / (c+d)
    E2 = d * (a+b) / (c+d)

    LL = 2 * ((a * np.log(a/E1)) + (b * np.log(b/E2)))
    
    return LL


def calculate_p_value(LL_Abs):
    if LL_Abs >= 15.13:
        return 0.0001
    if LL_Abs >= 10.83:
        return 0.001
    if LL_Abs >= 6.63:
        return 0.01
    if LL_Abs >= 3.84:
        return 0.05
    return -1


def format_emotion_stats(emo_source, day='all', common=True):
    """
    Take word count information and spit out corpus statistics into a spreadsheet.
    """
    if emo_source not in ['afinn', 'emolex', 'opinion', 'pattern', 'swn']:
        raise Exception('-- Error: invalid emo_source ' + emo_source + ' choose "emolex" or "pattern".')

    if day == 'all':
        df_case = pd.read_csv(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/case_' + emo_source + '_cat_word_counts.txt', sep='\t')
        df_control = pd.read_csv(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/control_' + emo_source + '_cat_word_counts.txt', sep='\t')
    elif emo_source == 'pattern':
        df_case = pd.read_csv(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/case_pattern_day_' + str(day) + '_cat_word_counts.txt', sep='\t')
        df_control = pd.read_csv(BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/control_pattern_day_' + str(day) + '_cat_word_counts.txt', sep='\t')
    else:
        raise Exception('-- Per-day option not yet implemented for', emo_source)

    # calculated over entire corpus by whitesapce split
    case_tokens = 20187666 # from Z:/Andre Bittar/Projects/eHOST-IT/data/case_30_text_ordinal_dates_pattern_p2.pickle
    control_tokens = 41909800 # from Z:/Andre Bittar/Projects/eHOST-IT/data/control_30_text_ordinal_dates_pattern_p2.pickle
    cc_token_ratio = case_tokens / control_tokens

    df_case.columns = ['emotion', 'word', 'count_case', 'abs_freq_case', 'rel_freq_case']
    df_control.columns = ['emotion', 'word', 'count_control', 'abs_freq_control', 'rel_freq_control']

    # merging retains only the words that are in both data sets
    if common:
        df_cc = df_case.merge(df_control, on=['emotion', 'word'], how='inner')

        df_cc['norm_freq_ratio_case_control'] = df_cc.count_case / df_cc.count_control / cc_token_ratio
        df_cc['LL'] = calculate_log_likelihood(df_cc)
        df_cc['LL_Abs'] = df_cc.LL
        df_cc['p_value'] = df_cc['LL_Abs'].apply(calculate_p_value)
        # add negative sign to show term on average more frequent in control
        df_cc.loc[df_cc.norm_freq_ratio_case_control < 1, 'LL'] = df_cc.LL * -1
        
    else:
        df_cc = df_case.merge(df_control, on=['emotion', 'word'], how='outer', indicator=True)
        df_cc = df_cc.loc[df_cc._merge.isin(['right_only', 'left_only'])]
        df_cc.drop('_merge', axis=1, inplace=True)
    
    for col in df_cc:
        dt = df_cc[col].dtype
        if dt == int or dt == float:
            df_cc[col].fillna(value=0, inplace=True)
    
    if common:
        if day == 'all':
            pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/cc_' + emo_source + '_stats.xlsx'
        else:
            pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/cc_' + emo_source + '_day_' + str(day) + '_stats.xlsx'
        df_cc.to_excel(pout)
        print('-- Wrote stats file:', pout)
    else:
        if day == 'all':
            pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/cc_' + emo_source + '_diff_stats.xlsx'
        else:
            pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/cc_' + emo_source + '_day_' + str(day)+ '_diff_stats.xlsx'
        df_cc.to_excel(pout)
        print('-- Wrote stats file:', pout)

    case_zero = len(df_cc[['word', 'count_case']].drop_duplicates().loc[df_cc.count_case == 0])
    control_zero = len(df_cc[['word', 'count_control']].drop_duplicates().loc[df_cc.count_control == 0])
    
    print()
    print('Global statistics:')
    print('-- Common words only:', common)
    print('-- Total', emo_source, 'words found:', len(df_cc.word.unique()))
    print('-- Words in control, but not in case:', case_zero)
    print('-- Words in case, but not in control:', control_zero)

    return df_cc


def aggregate_words(ctype, emo_source, day='all', control_stop_index=None):
    """
    Count the occurrences of words matched by Pattern/Afinn/Opinion etc.
    control_stop_index=114504 to get same number of tokens as cases
    """
    print('-- Aggregating ' + emo_source + ' words for', ctype)
    
    fin = BASE_DIR_Z + 'data/case_30_text_ordinal_dates_' + emo_source + '_p2.pickle'
    
    if ctype == 'control':
        fin = BASE_DIR_Z + 'data/control_30_text_ordinal_dates_' + emo_source + '_p2.pickle'

    elif ctype != 'case':
        raise Exception('-- Error: invalid ctype ' + ctype + ' choose "case" or "control".')                
    
    neg_words = []
    pos_words = []
    df = pd.read_pickle(fin)

    if day != 'all' and (isinstance(day, int) or isinstance(day, float)):
        print('-- Using only rows for day', day)
        df = df.loc[df.Date_ord_norm == day]

    for i, row in df.iterrows():
        if isinstance(row.neg_words, list):
            neg_words.extend(row.neg_words)
        if isinstance(row.pos_words, list):
            pos_words.extend(row.pos_words)
        # stop at as specified index in the control data in order to ensure we are looking
        # at the same number of tokens for cases and controls (see get_same_number_of_tokens)
        # here we stop immediately after the specified row as they are in ascending order
        if control_stop_index is not None and i == control_stop_index:
            break
    
    neg_counts = Counter(neg_words)
    pos_counts = Counter(pos_words)
    
    d = {'positive': pos_counts, 'negative': neg_counts}
    
    mdir = BASE_DIR_Z + 'data/emotion/' + emo_source + '_output/'
    if not os.path.exists(mdir):
        os.makedirs(mdir)
    pout = mdir + ctype + '_' + emo_source + '_cat_word_counts.pickle'
    pickle.dump(d, open(pout, 'wb'))
    print('-- Wrote', ctype, 'counts for', emo_source, 'to', pout)

    return d


def aggregate_emotion_words(ctype, emotion_source, method, token_match_on=None, control_stop_index=None):
    """
    Aggregate all emotion words.
    control_stop_index=114504 for same number of tokens as cases
    """
    print('-- Aggregating emotion words in', emotion_source , 'for', ctype)

    if token_match_on in ['word', 'lemma']:
        mdir = BASE_DIR_Z + 'data/emotion/' + emotion_source + '_' + method + '_' + token_match_on + '_offsets/' + ctype
    elif token_match_on is None:
        mdir = BASE_DIR_Z + 'data/emotion/' + emotion_source + '_' + method + '_offsets/' + ctype
    else:
        raise Exception('-- Error: invalid token_match_on ' + token_match_on + ' choose "word" or "lemma".')

    if ctype not in  ['case', 'control']:
        raise Exception('-- Error: invalid ctype ' + ctype + ' choose "case" or "control".')
    elif ctype == 'case' and control_stop_index is not None:
        print('-- Warning: parameter control_stop_index has no effect with cases.')
    
    files = [f for f in os.listdir(mdir) if f.startswith(ctype) and f.endswith('pickle')]
    
    if ctype == 'control':
        print('-- Filtering files...')
        print('-- Initial number of files:', len(files))
        files_filtered = []
        for f in files:
            # stop at as specified index in the control data in order to ensure we are looking
            # at the same number of tokens for cases and controls (see get_same_number_of_tokens)
            # here we skip any files that have an index above the specified value as we
            # cannot be sure the files will be in the same order as rows in the original data frame
            if control_stop_index is not None:
                index = int(f.split('.')[0].split('_')[-1])
                #index = int(f.replace('.pickle', '').split('_')[-1])
                if index > control_stop_index:
                    #print('-- Skipping control file:', f)
                    continue
            files_filtered.append(f)
        files = files_filtered
        print('-- Final number of files  :', len(files))
        del files_filtered
    
    cat_dict = {}
    n = 0

    for f in files:
        pin = os.path.join(mdir, f)
        n += 1
        if n % 1000 == 0:
            print(n, pin)
        d = pd.read_pickle(pin)
        for cat in d:
            c = cat_dict.get(cat, Counter())
            c = c + Counter([t[0] for t in d[cat]])
            cat_dict[cat] = c

    return cat_dict, files


def aggregate_corpus_words(ctype):
    """
    Collect statistics for a given sub-corpus, i.e. case or control
    """
    
    if ctype == 'case':
        df = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/case_30_text_pp.pickle')
    elif ctype == 'control':
        df = pd.read_pickle(BASE_DIR_Z + 'data/cc_text_preprocessed/control_30_text_pp.pickle')

    # remove all whitesapce tokens (kept by spaCy)
    df = df['tokens_' + ctype].apply(lambda x: [t for t in x if not t.endswith('__SP')])
    
    # only retain the word token (remove lemma and POS)
    df = df.apply(lambda x: [t.split('_')[0] for t in x])

    # define function to flatten the list of lists (of tokens)
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    words = flatten(df.tolist())
    
    fd = FreqDist(words)
    
    # remove stopwords
    for w in stopwords.words('english'):
        del fd[w]
    
    # remove punctuation tokens etc.
    new_fd = FreqDist()
    for c in fd:
        if re.search('^[\.,:;\?\!"\(\)\-\–_\{\}\[\]<>\^\|\\// \n\t\*#“”…\']+$', c) is None:
            new_fd[c] = fd[c]
    del new_fd[''] # remove weird blank character token
    
    # TODO see if I can use nltk's probability distributions to do some fancy calculations
    # http://www.nltk.org/_modules/nltk/probability.html
    
    return new_fd


def load_extracted_words_to_dataframe(ctype, emo_source, method, token_match_on=None):
    """
    Load the words extracted and stored with their offsets into a DataFrame.
    Do this after batch_get_offsets.
    """
    print('-- Loading extracted emotion words from', emo_source, 'for', ctype)
    
    mdir = None
    if token_match_on is not None:
        mdir = BASE_DIR_Z + 'data/emotion/' + emo_source + '_' + method + '_' + token_match_on + '_offsets/' + ctype
    else:
        mdir = BASE_DIR_Z + 'data/emotion/' + emo_source + '_' + method + '_offsets/' + ctype

    files = [f for f in os.listdir(mdir) if f.startswith(ctype) and f.endswith('pickle')]

    cols = ['pos_words_' + emo_source, 'neg_words_' + emo_source, 'pos_' + emo_source, 'neg_' + emo_source]
    
    if emo_source == 'emolex':
        cols += ['anger_emolex', 'anticipation_emolex', 
                 'disgust_emolex', 'fear_emolex', 
                 'joy_emolex', 'sadness_emolex', 
                 'surprise_emolex', 'trust_emolex']
        cols += ['anger_words_emolex', 'anticipation_words_emolex', 
                 'disgust_words_emolex', 'fear_words_emolex', 
                 'joy_words_emolex', 'sadness_words_emolex', 
                 'surprise_words_emolex', 'trust_words_emolex']

    n = 0
    df = pd.DataFrame()
    t = len(files)
    for f in files:
        pin = os.path.join(mdir, f)
        #print(pin)
        if n % 1000 == 0:
            print(n, '/', t)
        n += 1       
        # extract attributes from file name
        #case_10000000pk2006-07-26_157240311257233_0
        attrs = f.split('_')
        pk = attrs[1]
        cndocid = attrs[2]
        brcid, admidate = pk.split('pk')
        i = int(attrs[3].split('.')[0])
        
        d = pd.read_pickle(pin)
        df_tmp1 = pd.DataFrame([{k + '_words_' + emo_source: [item for sublist in v for item in sublist if isinstance(item, str)] for (k, v) in d.items()}], index=[i])
        df_tmp2 = pd.DataFrame({'pk': pk, 'brcid_' + ctype: brcid, 'CN_Doc_ID': cndocid, 'admidate': admidate}, index=[i])
        df_tmp = pd.concat([df_tmp2, df_tmp1], axis=1, sort=False)

        df = df.append(df_tmp, ignore_index=False, sort=False) #, sort=True)

    # do this for the sake of uniformity with other outputs
    df.rename(columns={'negative_words_' + emo_source: 'neg_words_' + emo_source, 'positive_words_' + emo_source: 'pos_words_' + emo_source}, inplace=True)
    
    # add columns if they don't exist
    for col in set(cols).difference(set(df.columns.unique())):
        df[col] = [[] for _ in range(len(df))]
    
    # also replace NaN with []
    for col in [c for c in cols]:
        inds = df.loc[pd.isna(df[col]), :].index
        for i in inds:
            df.at[i, col] = []
        
    # sentiment scores are just the number of words in each column
    for col in [c for c in cols if c.endswith('words_' + emo_source)]:
        df[col.replace('_words', '')] = df[col].apply(len)
 
    # set types (for merging later with DataFrame that contains ordinal day)
    df['admidate'] = pd.to_datetime(df.admidate)
    df['brcid_' + ctype] = df['brcid_' + ctype].astype(np.int64)
    
    pout = mdir + '_extracted_' + emo_source + '.pickle'
    print('-- Saving to file:', pout)
    df.to_pickle(pout)
    
    return df


def merge_extracted_with_per_day(ctype, emo_source, method, token_match_on=None):
    """
    Merge extracted words (e.g. result of load_extracted_words_to_dataframe())
    with "original" data (i.e. brcid, dates, text, etc.)
    """
    print('-- Merging extracted emotion words from', emo_source, 'for', ctype, ' with main data.')
    # TODO generalise this for all lexicons
    df_extracted = None
    if token_match_on is not None:
        df_extracted = pd.read_pickle(BASE_DIR_Z + 'data/emotion/' + emo_source + '_' + method + '_' + token_match_on + '_offsets/' + ctype + '_extracted_' + emo_source + '.pickle')
    else:
        df_extracted = pd.read_pickle(BASE_DIR_Z + 'data/emotion/' + emo_source + '_' + method + '_offsets/' + ctype + '_extracted_' + emo_source + '.pickle')

    df = pd.read_pickle(BASE_DIR_Z + 'data/' + ctype + '_30_text_ordinal_dates_p2.pickle')

    # set types (for merging later with DataFrame that contains ordinal day)
    missing_indexes = df.index.difference(df_extracted.index)
    word_columns = [w for w in df_extracted.columns if 'words' in w]
    
    df_extracted['admidate'] = pd.to_datetime(df_extracted.admidate)
    df_extracted['brcid_' + ctype] = df_extracted['brcid_' + ctype].astype(np.int64)
    
    df_extracted = df_extracted.reindex(df.index)
    
    df_merged = df[['pk', 'brcid_' + ctype, 'CN_Doc_ID', 'Date_ord_norm']].merge(df_extracted, on=['pk', 'brcid_' + ctype, 'CN_Doc_ID'], how='left', left_index=True, right_index=True)
    df_merged = pd.concat([df_merged, pd.DataFrame(df['text_' + ctype])], axis=1)
    df_merged.fillna(value=0.0, inplace=True)
    
    n = 0
    
    # set correct values
    for i in missing_indexes:
        for w in word_columns:
            df_merged.at[i, w] = [] # TODO check this - it doesn't seem to be working as values are still 0
        if df_merged.iloc[i].admidate == 0.0:
            df_merged.at[i, 'admidate'] = df.iloc[i].admidate
        if n % 1000 == 0:
            print(n, '/', len(missing_indexes))
        n += 1
    
    # check if this is needed for all data
    df_merged.drop(index=df_merged.last_valid_index(), inplace=True)
    
    pout = None
    if token_match_on is not None:
        pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_' + method + '_' + token_match_on  + '_offsets/' + ctype + '_30_text_merged_'  + emo_source + '_p2.pickle'
    else:
        pout = BASE_DIR_Z + 'data/emotion/' + emo_source + '_' + method + '_offsets/' + ctype + '_30_text_merged_'  + emo_source + '_p2.pickle'        
    
    df_merged.to_pickle(pout)
    
    print('-- Wrote file:', pout)
    
    return df_merged


def do_final_sequence():
    """
    Run the final process.
    """
    t0 = time()
    for c in ['case', 'control']:
        for e in ['afinn', 'emolex', 'opinion', 'pattern', 'swn']:
            _ = aggregate_words(c, e)
            get_emotion_stats(c, e)
    
    for e in ['afinn', 'emolex', 'opinion', 'pattern', 'swn']:
        _ = format_emotion_stats(e)
    t1 = time()
    print('-- Total time:', t1 - t0)


def check_in_lexicons(path, check_words=[]):
    """
    Check whether the list of new seed words provided by Rina are in a lexicon.
    """
    if len(check_words) < 0:
        # seed words
        check_words = ['pleasant', 'unfortunate', 'confident', 'doubt', 'hope', \
        'difficult', 'glad', 'challenging', 'trust', 'unsure', 'positive', \
        'negative']

    liwc = 'liwc' in path.lower()
    swn = 'senti' in path.lower()

    if swn:
        df = filter_sentiwordnet(use_mwe=False)
    else:
        df = load_lexicon(path)

    total = len(check_words)
    matched = []

    for s in check_words:
        words = df.loc[df.word.str.startswith(s[0])]

        for i, row in words.iterrows():
            word = row.word

            if liwc:
                if re.search(word, s, flags=re.I) is not None:
                    print(s, word)
                    matched.append(s)
            else:
                if word == s:
                    emo = df.at[i, 'emotion']
                    print(s, word, emo)
                    matched.append(s)

    print('matched:', len(matched), 'total:', total, 'missing:', total - len(matched), 'missing words:', sorted(set(check_words).difference(set(matched))))

    return df


if __name__ == '__main__':
    # text = 'The patient is very happy, but she is not depressed and is elated.'
    #df_lex = load_lexicon()
    #df_case = pd.read_pickle(BASE_DIR_Z + 'data/case_30_text_ordinal_dates_pattern_p2.pickle')
    
    #for i, row in df_case.iterrows():
    #    a = get_offsets_by_token(df_lex, row.text_case)
    #    print(i, a, row.text_case, len(row.text_case))
    #    for e in a:
    #        for k in a[e]:
    #            print(e + '\t' + row.text_case[k[0]:k[1]])
    #    break
    #batch_get_offsets(case=False, control=False)
    #df_swn = filter_sentiwordnet()
    #batch_get_offsets(RESOURCE_DIR + 'McL_keywords_mood.txt', 'mcl_mood')
    #batch_get_offsets(RESOURCE_DIR + 'McL_mwe_keywords.txt', 'mcl_mwe',  case=False)
    #batch_get_offsets(RESOURCE_DIR + 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'emolex', 'token', token_match_on='word', case=True, control=True)

    # this was to test time
    # EmoLex:
    # token-word: 0.01799250000052982 <-- nearly 3 times faster
    # regex     : 0.052834166685740154
    # Opinion
    # token-word: 0.014128888845443726
    # regex     : 0.039539444446563723
    # TODO: batch_get_offsets(RESOURCE_DIR + 'AFINN-en-165-8-emoticons_combined.txt', 'afinn', 'token', token_match_on='word', case=True, control=True)
    # TODO: batch_get_offsets(RESOURCE_DIR + 'opinion_pos_neg_combined.txt', 'opinion', 'token', token_match_on='word', case=True, control=True)
    
    #batch_get_offsets(RESOURCE_DIR + 'AFINN-en-165-8-emoticons_combined.txt', 'afinn', 'token', token_match_on='word', case=False, control=True)
    #batch_get_offsets(RESOURCE_DIR + 'opinion_pos_neg_combined.txt', 'opinion', 'token', token_match_on='word', case=True, control=True)
    
    if False:
        from time import time
    
        t0 = time()
        batch_get_offsets(RESOURCE_DIR + 'opinion_pos_neg_combined.txt', 'opinion', 'token', token_match_on='word', case=True, control=False)
        t1 = time()
    
        t2 = time()
        batch_get_offsets(RESOURCE_DIR + 'opinion_pos_neg_combined.txt', 'opinion', 'regex', case=True, control=False)
        t3 = time()
        
        first = (t1 - t0) / 3600
        second = (t3 - t2) / 3600
        print('token-word:', first)
        print('regex     :', second)
        print('fastest   :', end='')
        if first > second:
            print('token-word')
        else:
            print('regex')
            