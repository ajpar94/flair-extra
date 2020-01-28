#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script for selecting hyperparameter (ner)
#
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings on the Performance
#       of Intent-Detection and Named Entity Recognition for German texts'
#
#
# @usage: python hyperparameter_search.py /path/to/ner_folder /path/to/embedding /path/to/destination/


import argparse
from flair.datasets import ColumnCorpus
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.hyperparameter.param_selection import SequenceTaggerParamSelector, OptimizationValue
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

if __name__ == "__main__":
    # parser configuration
    parser = argparse.ArgumentParser(description='Script for selecting hyperparameter (ner)')
    parser.add_argument('ner_folder', type=str, help='ner_folder')
    parser.add_argument('embedding', type=str, help='embedding')
    parser.add_argument('dst', type=str, help='destination folder')
    args = parser.parse_args()

    # Prepare Corpus
    # define columns
    columns = {0: 'text', 1: 'ner'}
    # this is the folder in which train, test and dev files reside
    data_folder = args.ner_folder
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus = ColumnCorpus(data_folder, columns, train_file='train.txt', test_file='test.txt', dev_file='dev.txt')
    print(corpus)
    # 2. what tag do we want to predict?
    tag_type = 'ner'
    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    # Language models
    lm_fwd = args.embedding + "_fwd/best-lm.pt"
    lm_bwd = args.embedding + "_bwd/best-lm.pt"

    # define your search space
    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[StackedEmbeddings([FlairEmbeddings(lm_fwd), FlairEmbeddings(lm_bwd)])])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[512, 1024])
    search_space.add(Parameter.ANNEAL_FACTOR, hp.choice, options=[0.5, 0.75])
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.1, 0.5, 1.0])
    search_space.add(Parameter.PATIENCE, hp.choice, options=[3, 5, 7])
    search_space.add(Parameter.DROPOUT, hp.choice, options=[0.15])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8])

    # create the parameter selector
    param_selector = SequenceTaggerParamSelector(corpus, 'ner', args.dst, max_epochs=42, training_runs=1, optimization_value=OptimizationValue.DEV_SCORE)

    # start the optimization
    param_selector.optimize(search_space, max_evals=40)
