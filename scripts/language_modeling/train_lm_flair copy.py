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
# import importlib
import json
import logging
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

if __name__ == "__main__":
    # parser configuration
    parser = argparse.ArgumentParser(description='Script for FLAIR LM training')
    parser.add_argument('-c', '--corpus_path', type=str, help='path to corpus')
    parser.add_argument('-t', '--train_path', type=str, help='path to model,logs and checkpoints')
    parser.add_argument('-o', '--options_file', type=str, help='file with parameters')
    parser.add_argument('--continue_training', action='store_true', help='continue from checkpoint?')
    args = parser.parse_args()

    # logging config
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    log = logging.getLogger(__name__)

    with open(args.options_file) as json_file:
        options = json.load(json_file)

    '''
    # import options
    try:
        options = importlib.import_module(args.options_file).options
    except ImportError as err:
        print('Error:', err)
    '''

    # load the default character dictionary
    # TODO: add possibility for other dictionary!
    # (https://github.com/zalandoresearch/flair/issues/179#issuecomment-433942853)
    print("loading Dictionary")
    dictionary = Dictionary.load('chars')
    # instantiate corpus
    log.info("Making corpus from folder: {}".format(args.corpus_path))
    corpus = TextCorpus(args.corpus_path,
                        dictionary,
                        options['is_forward_lm'],
                        **options['corpus'])

    # TRAINING
    if args.continue_training:
        # load checkpoint
        cp_path = args.train_path + '/checkpoint.pt'
        log.info("Continue training from {}".format(cp_path))
        # load LM-Trainer
        trainer = LanguageModelTrainer.load_from_checkpoint(cp_path, corpus)
    else:
        # instantiate language model
        log.info("Creating language model")
        language_model = LanguageModel(dictionary,
                                       options['is_forward_lm'],
                                       **options['language_model'])
        # instantiate LM Trainer
        trainer = LanguageModelTrainer(language_model, corpus)

    log.info("Starting training. See {}".format(args.train_path))
    trainer.log_interval = 500
    trainer.train(args.train_path, **options['training'])
