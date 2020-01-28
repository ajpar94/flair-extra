# flair-extra

This repository is mainly a collection of python-scripts that simplify some of the workflows when using [flair](https://github.com/flairNLP/flair):
* **Preprocessing, formatting and analyzing text datasets**
* **Training embeddings / language models (from scratch)**
* **Performing Named-Entity-Recognition and Intent-Detection**
* **Evaluating trained models**

## Example
Say you have a raw text corpus and a NER dataset from the same domain. You want to train your own flair language model in order to use it for training a NER model for your domain. Steps in that workflow include:
* Preprocess, format and analyze your corpus/datasets
* Train a custom language model (LM)
* Train an NER model (with your previously trained LM, maybe even in combination with other LMs/embeddings)
* Evaluate the performance of yout final model

### Preproccess
e.g. clean, replace umlaute, remove accents and puntuaction tokens
```bash
$ cd scripts/language_modeling
$ python preprocess_corpus.py -cuap /path/to/corpus/ /path/to/corpus_proc/
```

### Format
e.g. create a corpus folder from a plain text file using 97% as training-, 1% as validation and 2% as test set. Split training set into 20 parts
```bash
$ cd scripts/language_modeling
$ python make_corpus_folder.py /path/to/corpus/ /path/to/corpus/folder/ -p 97-1-2 -s 20
```

e.g. create a column corpus folder (NER) from a column file using 70% as training-, 20% as validation and 10% as test set. Shuffle lines
```bash
$ cd scripts/named_entity_recognition
$ python make_nercorpus_folder.py /path/to/ner_dataset /path/to/destination/ -p 70-20-10 --shuffle
```
