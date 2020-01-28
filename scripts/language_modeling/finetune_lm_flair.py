#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script for training a FLAIR language model
#
# @author: Ajit Parikh
# @see: Master Thesis 'Effects of different Word Embeddings on the Performance
#       of Intent-Detection and Named Entity Recognition for German texts'
#
#
# @usage:
# Set parameters in options.py
# >>> python train_lm_flair.py -c /path/to/corpus/ -t /path/to/model/folder/ -o options_lm_flair [--continue_training]

import sys
import argparse
import importlib
from pathlib import Path
from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

if __name__ == "__main__":
    # parser configuration
    parser = argparse.ArgumentParser(description='Script for FLAIR LM training')
    parser.add_argument('-p', '--pretrained_model', type=str, help='pretrained flair model')
    parser.add_argument('-c', '--corpus_path', type=str, help='path to corpus')
    parser.add_argument('-m', '--model_path', type=str, help='path to model,logs and checkpoints')
    parser.add_argument('-o', '--options_file', type=str, help='file with parameters')
    args = parser.parse_args()

    # import options
    try:
        options = importlib.import_module(args.options_file).options
    except ImportError as err:
        print('Error:', err)

    # instantiate an existing LM, such as one from the FlairEmbeddings
    language_model = FlairEmbeddings(args.pretrained_model).lm

    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm

    # get the dictionary from the existing language model
    dictionary = language_model.dictionary

    # instantiate corpus
    corpus = TextCorpus(Path(args.corpus_path),
                        dictionary,
                        is_forward_lm,
                        **options['corpus'])

    # use the model trainer to fine-tune this model on your corpus
    trainer = LanguageModelTrainer(language_model, corpus)
    trainer.log_interval = 500

    trainer.train(Path(args.model_path), **options['training'])
