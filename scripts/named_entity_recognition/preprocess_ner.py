#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to preprocess for NER file
#
# @author: Ajit Parikh
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings on the Performance
#       of Intent-Detection and Named Entity Recognition for German texts'
#
#
# @usagee: python preprocess_ner.py /path/to/ner_dataset /path/to/ner_datset_pp [--lemma] [-stem]

import os
from smart_open import open
import numpy as np
import pandas as pd
import argparse
import spacy
from nltk.stem.snowball import GermanStemmer


def write_to_csv(df_list):
    """
    Writes multiple dataframes as csv to a file
    with blank lines inbetween
    """
    filePath = args.dst
    if os.path.exists(filePath):
        print("deleting", filePath)
        os.remove(filePath)

    with open(filePath, 'a') as f:
        for df in df_list:
            df.to_csv(f, sep="\t", index=False, header=False)
            f.write("\n")

    print("created", filePath, "with", len(df_list), "entries.")


if __name__ == "__main__":
    # parse config
    parser = argparse.ArgumentParser(description='Script for preprocessing NER dataset')
    parser.add_argument('src', type=str, help='source file')
    parser.add_argument('dst', type=str, help='destination file')
    parser.add_argument('--lemma', action='store_true', help='lemmatize')
    parser.add_argument('--stem', action='store_true', help='lemmatize')
    args = parser.parse_args()

    if args.lemma:
        # python -m spacy download de_core_news_sm
        nlp = spacy.load('de_core_news_sm')

    if args.stem:
        stemmer = GermanStemmer()

    df = pd.read_csv(args.src, sep='\t', header=None, skip_blank_lines=False, names=['token', 'tag'])
    df_list = np.split(df, df[df.isnull().all(1)].index)
    df_list = [df.dropna() for df in df_list if not df.dropna().empty]

    df_list_new = []

    for df in df_list:
        if args.lemma:
            sentence = ' '.join(df['token'].values)
            doc = nlp(sentence)
            tokens = [token.lemma_ for token in doc]
        else:
            tokens = df['token'].values
        if args.stem:
            tokens = [stemmer.stem(t) for t in tokens]

        df['token'] = tokens
        df_list_new.append(df)

    write_to_csv(df_list_new)
