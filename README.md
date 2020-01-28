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

The following examples assume that you are familiar with flair base classes, data types, etc. You can get started with flair tutorials [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_1_BASICS.md). Most parts of this workflow (with output) are illustrated as [jupyter notebooks](notebooks/).

### Preprocess
e.g. clean, replace umlaute, remove accents and puntuaction tokens
```bash
$ cd scripts/language_modeling
$ python preprocess_corpus.py -cuap /path/to/corpus/ /path/to/corpus_proc/
```

### Format
E.g. create a corpus folder from a plain text file using 97% as training-, 1% as validation and 2% as test set. Split training set into 20 parts
```bash
$ cd scripts/language_modeling
$ python make_corpus_folder.py /path/to/corpus/ /path/to/corpus/folder/ -p 97-1-2 -s 20
```

E.g. create a column corpus folder (NER) from a column file using 70% as training-, 20% as validation and 10% as test set. Shuffle lines
```bash
$ cd scripts/named_entity_recognition
$ python make_nercorpus_folder.py /path/to/ner_dataset /path/to/destination/ -p 70-20-10 --shuffle
```

### Analyze
Analyze text corpus: lines, tokens, etc.; most common tokens; wordcloud
```python
from modules.corpus_analysis import TextAnalysis

ta = TextAnalysis(path/to/corpus)

print(ta.obtain_statistics())
print(ta.most_common_tokens(nr_tokens=15, stop_words=['the', 'is',...]))
ta.wordcloud(stop_words=[], savefig_file=None, figsize = ((15,15)))
```

Analyze a ColumnCorpus (NER): tags/tokens stats; most common tokens; wordcloud; visualize sentences, tag distribution, most common tokens per tags
```python
from modules.corpus_analysis import ColumnCorpusAnalysis

columns = {0: 'text', 1: 'ner', 2: 'pos'}
cca = ColumnCorpusAnalysis(path=path/to/corpus_folder, columns=columns, tag_types=['ner', 'pos'])

print(cca.obtain_statistics(tag_type='ner'))
print(cca.most_common_tokens(nr_tokens=20, stop_words=stopwords.words('german')))
cca.wordcloud(savefig_file=None, figsize = ((15,15))
cca.visualize_ner_tags(display_index=range(5))
cca.tag_distribution(savefig_file=None, tag_type='ner', figsize=(13, 10))
cca.most_common_tokens_per_tags(max_tokens=10, tag_type='ner', print_without_count=True)
```

### Train a Flair Language Model
First, set parameters (epochs, learning rate, etc.) in a special file (e.g. options_lm.json). Then:
```bash
$ cd scripts/language_modeling
$ python train_lm_flair.py -c /path/to/corpus/ -t /path/to/model/folder/ -o options_lm.json [--continue_training]
```

### Train a NER Model
Again, set parameters in a options-file. Say, you have trained a forward and a backwards LM stored in fwd-lm.pt and bwd-lm.pt, then:
```bash
$ cd scripts/named_entity_recognition
$ python train_ner_flair.py -c /path/to/corpus/ -t /path/to/model/folder/ -o options_ner_flair [--continue_training] [--tensorboard] -e fwd-lm.pt bwd-lm.pt
```
It will use a stacked combination of all embeddings you specify after te flag -e.

### Evaluate trained Model
Predictions; Precision, Recall, F1 scores (total/ for each tag) as table and as plot; training curves, weights, etc.
```python
from modules.model_evaluation import SequenceTaggerEvaluation

ste = SequenceTaggerEvaluation(path/to/model/folder/, model='best-model.pt')

text = "Lorem Ipsum ..."
sentences = SequenceTaggerEvaluation._preprocess(text)
ste.predict(sentences)

ste.result_tables()
ste.plot_tag_stats(mode='tp_fn',...)
ste.plot_training_curves()
ste.plot_weights()
ste.plot_learning_rate()
```

---
The code in this repository was used for the thesis: [*"Effects of different Word Embeddings on the Performance of Intent Detection and Named Entity Recognition for German texts"* (Parikh, 2019)](https://drive.google.com/file/d/1SEnWTUDwsD7_1ZN7e_ynmBys2pmfPNtK/view?usp=sharing).
