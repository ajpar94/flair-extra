#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to make a vocabalury file
#
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings on the Performance
#       of Intent-Detection and Named Entity Recognition for German texts'
#
#
# @example: python make_vocabulary.py /path/to/corpus/ /path/to/vocabulary-file [--top]


from smart_open import open
import argparse
from nltk.tokenize.toktok import ToktokTokenizer
from collections import defaultdict
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to make a vocabalury file')
    parser.add_argument('src', type=str, help='source file')
    parser.add_argument('dst', type=str, help='destination file')
    parser.add_argument('--top', type=int, default=10, help='print top x most frequent tokens with their count')
    args = parser.parse_args()

    tt = ToktokTokenizer()
    vocab = defaultdict(int)

    with open(args.src, 'r') as fin:
        for i, line in enumerate(tqdm(fin)):
            tokens = tt.tokenize(line)
            for t in tokens:
                vocab[t] += 1

    with open(args.dst, 'w') as fout:
        fout.write("</S>\n<S>\n<UNK>")
        print("\n--", str(args.top), "most frequent tokens --")
        for i, v in enumerate(sorted(vocab, key=vocab.get, reverse=True)):
            if i < args.top:
                print(v + '\t\t' + str(vocab[v]))
            fout.write("\n" + v)
