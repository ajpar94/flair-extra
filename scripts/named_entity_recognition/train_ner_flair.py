#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script for training a NER model (Sequence Tagger)
#
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings on the Performance
#       of Intent-Detection and Named Entity Recognition for German texts'
#
#
# @usage:
# Set parameters in options.py
# >>> python train_ner_flair.py -c /path/to/corpus/ -t /path/to/model/folder/ -o options_ner_flair [--continue_training] [--tensorboard] -e fwd-lm.pt bwd-lm.pt

import argparse
import importlib
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


if __name__ == "__main__":
    # parser configuration
    parser = argparse.ArgumentParser(description='Script for FLAIR NER training')
    parser.add_argument('-c', '--corpus_path', type=str, help='path to corpus')
    parser.add_argument('-t', '--train_path', type=str, help='path to model,logs and checkpoints')
    parser.add_argument('-o', '--options_file', type=str, help='file with parameters')
    parser.add_argument('--continue_training', action='store_true', help='continue from checkpoint?')
    parser.add_argument('--tensorboard', action='store_true', help='use tensorboard?')
    parser.add_argument('-e', '--embeddings_list', nargs='+', default=[])
    args = parser.parse_args()

    # import options
    try:
        options = importlib.import_module(args.options_file).options
    except ImportError as err:
        print('Error:', err)

    # define columns
    columns = {0: 'text', 1: 'ner'}
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus = ColumnCorpus(args.corpus_path, columns,
                          train_file='train.txt',
                          test_file='test.txt',
                          dev_file='dev.txt',
                          **options['corpus'])
    print(corpus)

    # what tag to predict
    tag_type = 'ner'
    # make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    # embeddings
    if len(args.embeddings_list) == 1:
        embeddings = FlairEmbeddings(args.embeddings_list[0])
    else:
        embeddings = StackedEmbeddings([FlairEmbeddings(lm) for lm in args.embeddings_list])
    # initialize tagger
    tagger = SequenceTagger(embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            **options['sequences_tagger'])
    # initialize trainer
    if args.continue_training:
        checkpoint = tagger.load_checkpoint(args.train_path + 'checkpoint.pt')
        trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)
    else:
        trainer = ModelTrainer(tagger, corpus, use_tensorboard=args.tensorboard)
    # training
    trainer.train(args.train_path, **options['training'])
