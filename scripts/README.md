# Scripts - Usage

## [Language Modeling](scripts/language_modeling/)

### [preprocess a text corpus](scripts/language_modeling/preprocess_corpus.py)
example: clean, replace umlaute, remove accents and puntuaction tokens
```bash
$ python preprocess_corpus.py -cuap /path/to/corpus/ /path/to/corpus_pp/
```

### [make corpus folder](scripts/language_modeling/make_corpus_folder.py)
example: create a corpus folder from a plain text file using 97% as training-, 1% as validation and 2% as test set. Split training set into 20 parts
```bash
$ python make_corpus_folder.py /path/to/corpus/ /path/to/corpus/folder/ -p 97-1-2 -s 20
```

### [make vocabulary file](scripts/language_modeling/make_vocabulary.py)
create a vocabulary file from a plain text file with one token per line, sorted by frequency (descending). --top 10 prints out the 10 most frequent tokens with their frequency.
```bash
$ python make_vocabulary.py /path/to/corpus/ /path/to/vocabulary-file [--top]
```

### [train flair embedding](scripts/language_modeling/train_lm_flair.py)
```bash
$ python train_lm_flair.py -c /path/to/corpus/ -m /path/to/model/folder/ -o options_lm_flair [--continue_training]
```

### [finetune flair embedding](scripts/language_modeling/finetune_lm_flair.py)
```bash
$ python finetune_lm_flair.py -p 'de-forward' -c /path/to/corpus/ -m /path/to/model/folder/ -o options_lm_flair
```

### [train word2vec (TODO)](scripts/language_modeling/train_word2vec.py)

### [train fasttext (TODO)](scripts/language_modeling/train_fasttext.py)


## [Named Entity Recognition](scripts/named_entity_recognition/)

### [preprocess a column corpus](scripts/named_entity_recognition/preprocess_ner.py)
TODO: add more preprocessing options
```bash
$ python preprocess_ner.py /path/to/ner_dataset /path/to/ner_datset_pp [--lemma] [-stem]
```

### [make ner corpus folder](scripts/named_entity_recognition/make_nercorpus_folder.py)
example: create a corpus folder from a column file using 70% as training-, 20% as validation and 10% as test set. Shuffle lines
```bash
$ python make_nercorpus_folder.py /path/to/ner_dataset /path/to/destination/ -p 70-20-10 --shuffle
```

### [hyperparameter search](scripts/named_entity_recognition/hyperparameter_search.py)
```bash
python hyperparameter_search.py /path/to/ner_folder /path/to/embedding /path/to/destination/
```

### [train sequence tagger (ner)](scripts/named_entity_recognition/train_ner_flair.py)
-e (embeddings): list of paths to flair embeddings wich will be stacked
```bash
$ python train_ner_flair.py -c /path/to/corpus/ -t /path/to/model/folder/ -o options_ner_flair [--continue_training] [--tensorboard] -e fwd-lm.pt bwd-lm.pt
```

### [train sequence tagger (ner) with cross validation (TODO)](scripts/named_entity_recognition/train_ner_flair_crossval.py)
see how to do it [here](tutorials/TRAINING_A_NER_MODEL.md#variant-2-k-fold-cross-validation).
```bash
$ python train_ner_flair_crossval.py -c /path/to/corpus/ -t /path/to/model/folder/ -o options_ner_flair -f 10 [--tensorboard] -e fwd-lm.pt bwd-lm.pt
```

## [classification (TODO)](scripts/classification)