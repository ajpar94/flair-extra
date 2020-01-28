#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to preprocess corpora for embedding training
#
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings on the Performance
#       of Intent-Detection and Named Entity Recognition for German texts'
#
#
# @usage: python preprocess_corpus.py -uacps /path/to/corpus/ /path/to/corpus_pp/

import os
import sys
from smart_open import open
from textacy import preprocess as pp
from nltk.corpus import stopwords
import argparse
import re
import logging
from tqdm import tqdm
import multiprocessing as mp
import spacy
from nltk.stem.snowball import GermanStemmer


def get_total(line_number):
    if line_number != -1:
        return line_number
    else:
        with open(args.src, 'r') as fin:
            for i, line in enumerate(fin):
                pass
            return i


def clean_text(text):
    text = re.sub('[^a-zA-Z0-9ßöäüÖÄÜ_.:,;?!()&@/€\- ]', "", text)
    text = pp.normalize_whitespace(text)

    return text


def replace_umlaute(text):
    """
    Replaces special german characters like
    'umlaute' and 'ß' in given text.

    :param text: text as str
    :return: manipulated text as str
    """
    text = text.replace('ä', 'ae')
    text = text.replace('Ä', 'Ae')
    text = text.replace('ö', 'oe')
    text = text.replace('Ö', 'Oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ü', 'Ue')
    text = text.replace('ß', 'ss')

    return text


def preprocess(line):
    """
    Pre processes the given line.

    :param line: line as str
    :return: preprocessed sentence(s)
    """
    result = ''
    if len(line) < args.linelength:
        if args.clean:
            line = clean_text(line)
        if args.lemmatize:
            doc = nlp(line)
            tokens = [token.lemma_ for token in doc]
        else:
            tokens = line.split()
        if args.stem:
            tokens = [stemmer.stem(t) for t in tokens]
        if args.decapitalize:
            tokens = [t.lower() for t in tokens]
        if args.umlaute:
            tokens = [replace_umlaute(t) for t in tokens]
        if args.accents:
            tokens = [pp.remove_accents(t) for t in tokens]
        if args.numbers:
            tokens = [pp.replace_numbers(t, replace_with='*NUMMER*') for t in tokens]
        if args.punctuation:
            tokens = [t for t in tokens if t not in punctuation_tokens]
        if args.stopwords:
            tokens = [t for t in tokens if t.lower() not in stop_words]
        if args.forbidden:
            tokens = [t for t in tokens if not any(kw in t.lower() for kw in forbidden_keywords)]
        if len(tokens) > 3:
            result = "{}\n".format(' '.join(tokens))

    return result


# Main
if __name__ == "__main__":

    # parser configuration
    parser = argparse.ArgumentParser(description='Script for preprocessing corpora')
    parser.add_argument('src', type=str, help='source file with raw data')
    parser.add_argument('dst', type=str, help='destination file name to store processed corpus in')
    parser.add_argument('-l', '--lemmatize', action='store_true', help='lemmatize input')
    parser.add_argument('-s', '--stem', action='store_true', help='stem input')  # converts in lower case
    parser.add_argument('-d', '--decapitalize', action='store_true', help='convert everything to lowercase')
    parser.add_argument('-u', '--umlaute', action='store_true', help='replace german umlaute and ß')
    parser.add_argument('-a', '--accents', action='store_true', help='remove accents')
    parser.add_argument('-c', '--clean', action='store_true', help='clean text to only contain specific characters')
    parser.add_argument('-n', '--numbers', action='store_true', help='replace numbers with *NUMMER*')
    parser.add_argument('-p', '--punctuation', action='store_true', help='remove punctuation tokens')
    parser.add_argument('-f', '--forbidden', action='store_true', help='remove tokens that contain specified forbidden keywords')
    parser.add_argument('-w', '--stopwords', action='store_true', help='remove stop word tokens')
    parser.add_argument('--linelength', type=int, default=1000, help='filter/remove lines that are longer')
    parser.add_argument('--limit', type=int, default=-1, help='limit number of lines to be processed')
    parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='thread count')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for multiprocessing')
    args = parser.parse_args()

    # other configuration
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']',
                          '{', '}', '?', '!', '-', '–', '+', '*', '--', '---', '\'\'', '``']

    forbidden_keywords = ['@', 'http', 'www']

    # load german spaCy model
    if args.lemmatize:
        # python -m spacy download de_core_news_sm
        nlp = spacy.load('de_core_news_sm')

    if args.stem:
        stemmer = GermanStemmer()

    logging.info('Estimating processing time ...')
    total = get_total(args.limit)
    logging.info('Begin processing {} lines ...'.format(total))

    if not os.path.exists(os.path.dirname(args.dst)):
        os.makedirs(os.path.dirname(args.dst))

    if args.stopwords:
        if not args.umlaute:
            stop_words = stopwords.words('german')
        else:
            stop_words = [replace_umlaute(token) for token in stopwords.words('german')]

    with open(args.src, 'r') as fin:
        with open(args.dst, 'w') as fout:
            pool = mp.Pool(args.threads)
            values = pool.imap(preprocess, fin, chunksize=args.batch_size)
            pbar = tqdm(total=total)
            for i, s in enumerate(values):
                if i:
                    if i <= total:
                        if i % 25000 == 0:
                            # logging.info('processed {} sentences'.format(i))
                            fout.flush()
                        if s:
                            fout.write(s)
                            pbar.update()
                    else:
                        break
            pbar.n = total
            pbar.refresh()
            pbar.close()
            logging.info('preprocessing of {} lines finished!'.format(i))

    """
    # without multiprocessing
    with open(args.src, 'r') as fin:
        with open(args.dst, 'w') as fout:
            pbar = tqdm(total=total)
            for i, line in enumerate(fin):
                if i <= total:
                    if i % 25000 == 0:
                        # logging.info('processed {} sentences'.format(i))
                        fout.flush()
                    s = preprocess(line)
                    fout.write(s)
                    pbar.update()
                else:
                    break
            pbar.n = total
            pbar.refresh()
            pbar.close()
            logging.info('preprocessing of {} sentences finished!'.format(i))
    """
