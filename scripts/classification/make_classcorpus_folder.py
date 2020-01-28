#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to to build dataset folder for Intent-Detection
#
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings
#                      on the Performance of Intent-Detection
#                      and Named Entity Recognition for German texts'
#
#
# @example: python build_dataset_NER.py datasets/telecommication_manually_tagged_1k.txt /datasets/formatted/tmt_1k/ -p 70-15-15 --shuffle


import os
from smart_open import open
import numpy as np
import pandas as pd
import random
import argparse

'''
# parser configuration
parser = argparse.ArgumentParser(description='Script for building dataset folder for NER with train/test/dev split')
parser.add_argument('src', type=str, help='source file with raw data')
parser.add_argument('dst', type=str, help='destination folder')
parser.add_argument('-p', '--percentage', type=str, help='percentage for train-test-validation split')
parser.add_argument('--shuffle', action='store_true', help='shuffle dataset before splitting')
args = parser.parse_args()


def ttd_split(arr, q1, q2, shuffle=True):
    """
    Splits a list at given indices (percentages)
    """
    ls = arr.copy()
    if shuffle:
        random.shuffle(ls)
    ln = len(ls)
    fst = ls[:int(ln * q1)]
    snd = ls[int(ln * q1):int(ln * q2)]
    trd = ls[int(ln * q2):]

    return fst, snd, trd


def write_to_csv(df_list, file):
    """
    Writes multiple dataframes as csv to a file
    with blank lines inbetween
    """
    filePath = args.dst + file
    if os.path.exists(filePath):
        print("deleting", filePath)
        os.remove(filePath)

    with open(filePath, 'a') as f:
        for df in df_list:
            df.to_csv(f, sep="\t", index=False, header=False)
            f.write("\n")

    print("created", filePath, "with", len(df_list), "entries.")

# TODO! assert if paths exist!


# calculate percentages for splitting
q = list(map(int, args.percentage.split('-')))
assert sum(q) == 100, "percentages don't add up to 100%"
q1 = q[0] / 100
q2 = (q[0] + q[1]) / 100

# read dataset into dataframe
df = pd.read_csv(args.src, sep='\t', header=None, skip_blank_lines=False)
# split dataframe at blank lines
df_list = np.split(df, df[df.isnull().all(1)].index)
# remove blank lines
df_list = [df.dropna() for df in df_list if not df.dropna().empty]

print("The dataset contains", len(df_list), "entries.")

# split data
train, test, dev = ttd_split(df_list, q1, q2, shuffle=args.shuffle)

# write to respective files
write_to_csv(train, "train.txt")
write_to_csv(test, "test.txt")
write_to_csv(dev, "dev.txt")
'''
