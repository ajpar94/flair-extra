#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to make corpus folder
#
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings on the Performance
#       of Intent-Detection and Named Entity Recognition for German texts'
#
#
# @usage: python make_corpus_folder.py /path/to/corpus/ /dpath/to/corpus/folder/ -p 98-1-1 -s 20


import os
from smart_open import open
import argparse
import subprocess


def count_lines(file):
    with open(file, 'r') as fin:
        for i, line in enumerate(fin):
            pass
        print(file, "contains", i, "lines.")
        return i


def ttd_split(file, q1, q2):
    """
    Splits a file at given indices (percentages)
    """
    nlines = count_lines(file)
    t1 = int(nlines * q1)
    t2 = int(nlines * q2)

    with open(file, "r") as f:
        with open(args.dst + "train/train.txt", "w") as train:
            with open(args.dst + "valid.txt", "w") as valid:
                with open(args.dst + "test.txt", "w") as test:
                    for i, line in enumerate(f):
                        if i < t1:
                            train.write(line)
                        elif i < t2:
                            valid.write(line)
                        else:
                            test.write(line)

    split_lines = int(t1 / args.trainsplits) + 1
    subprocess.call(["split", "-l", str(split_lines), "train.txt", "train_"], cwd=args.dst + "train/")
    os.remove(args.dst + "train/train.txt")


if __name__ == "__main__":
    # parser configuration
    parser = argparse.ArgumentParser(description='Script for building corpus folder with train/test/dev split')
    parser.add_argument('src', type=str, help='source file with corpus')
    parser.add_argument('dst', type=str, help='destination folder')
    parser.add_argument('-p', '--percentage', type=str, help='percentage for train-test-validation split. ex. 80-10-10')
    parser.add_argument('-s', '--trainsplits', type=int, help='number of trainsplits')
    args = parser.parse_args()

    # calculate percentages for splitting
    q = list(map(int, args.percentage.split('-')))
    assert sum(q) == 100, "percentages don't add up to 100%"
    q1 = q[0] / 100
    q2 = (q[0] + q[1]) / 100

    os.makedirs(args.dst + "train/")

    ttd_split(args.src, q1, q2)
