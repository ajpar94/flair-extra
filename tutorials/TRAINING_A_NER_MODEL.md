
# Training a Sequence Labeling Model (Named-Entity-Recognition)

Example code for training a NER model with [flair](https://github.com/zalandoresearch/flair). See also [Tutorial: Training a Model](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).

This tutorial will show how to train a NER model with a **Train/Validation/Test split** and with **K-Fold Cross-Validation**.



### Google Colab setup 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajpar94/embeddings-comparison/blob/master/notebooks/train_ner_model.ipynb)

First, make sure to turn on 'Hardware accelerator: GPU' in *Edit > Notebook Settings*. Next, we will we mount our google drive to easily access corpora, datasets, embeddings and store models. Finally, install flair and configure your paths.


```python
# Mount google drive
from google.colab import drive
drive.mount('/gdrive')
```


```console
$ pip install flair --quiet
```


```python
# PATHS
from pathlib import Path

base_path = Path('/gdrive/My Drive/embeddings-comparison/resources')
emb_path = base_path/'models'/'embeddings'
ner_model_path = base_path/'models'/'taggers'
ner_corpus_path = base_path/'corpora'/'column_corpora'
```

---
## Variant 1 - Training/Validation/Test Split

### Sequence Labeling Dataset (Corpus)
A `ColumnCorpus` consists out of tagged sentences and is constructed by a file in column format where each line has one word together with its linguistic annotation. Sentences are seperated by blank line. Example:

```console
James B-person
Watson I-person
visited O
Germany B-country
in O
2019 B-year
. O

Sam B-person
was O
not B-negation
there O
. O
```

In our example the second column represents a ner tag in BIO format. You need three of those files: train, dev, test, which correspond to the training, validation and testing split during model training. You can also split one file by percentages using [*make_nercorpus_folder.py*](/scripts/named_entity_recognition/make_nercorpus_folder.py). For example, if you want use 5% of the sentences for validation, 10% for testing, 85% for training and also want shuffle, you can do:

```console
$ python build_dataset_ner.py columnfile.txt /path/to/output/ -p 85-5-10 --shuffle
```

Alternatively, use one of flair's prepared datasets. Define the `ColumnCorpus`, define what tag to predict and create a `tag_dictionary`. See also [this](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md#reading-your-own-sequence-labeling-dataset).







```python
# PREPARE CORPUS
# alternative: from flair.datasets import WIKINER_ENGLISH
from flair.datasets import ColumnCorpus

# define columns (multiple possible: ... 2: 'pos')
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
corpus_folder = ner_corpus_path/'example-corpus'

# init a corpus using column format, data folder 
# alternative: corpus = WIKINER_ENGLISH()
corpus = ColumnCorpus(corpus_folder, columns,
                      train_file='train.txt',
                      test_file='test.txt',
                      dev_file='dev.txt')
print(corpus)

# what tag do we want to predict?
tag_type = 'ner'

# make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
```

### Embeddings
flair comes with many embeddings out of the box (see: [Tutorial: List of All Word Embeddings](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)). Or point to your own custom embeddings. If you want to know how to train your own embeddings, check [Training a Flair Language Model](/tuorials/TRAINING_A_FLAIR_LM.md) and [Tutorial: Training your own Flair Embeddings](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).

`StackedEmbeddings` can be used combine multiple embeddings, which makes sense when you have a forward and a backward language model.


```python
# INITIALIZE EMBEDDINGS
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

# path to embeddings
lm_fwd = emb_path/'FLAIR'/'example-fwd'/'best-lm.pt'
lm_bwd = emb_path/'FLAIR'/'example-bwd'/'best-lm.pt'

embeddings = StackedEmbeddings([FlairEmbeddings(lm_fwd), FlairEmbeddings(lm_bwd)])
# alternative: embeddings = StackedEmbeddings([FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])
```

### Sequence Tagger (NER Model)
The `SequenceTagger` can take a lot more parameter (e.g. dropout). For a full list, check [here](https://github.com/zalandoresearch/flair/blob/master/flair/models/sequence_tagger_model.py#L68).


```python
# INITIALIZE SEQUENCE TAGGER
from flair.models import SequenceTagger

tagger = SequenceTagger(hidden_size=512,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type=tag_type,)
```

### Model Trainer
Define the path to the output/model folder. After training, this folder will usually contain:


*   final-model.pt
*   checkpoint.pt
*   weights.txt
*   loss.tsv
*   test.tsv
*   training.log

Depending on whether or not you `train_with_dev` there will a **best-model.pt** aswell. `ModelTrainer.train()` can take a lot of optional parameters. For a full list of parameters, check [here](https://github.com/zalandoresearch/flair/blob/master/flair/trainers/trainer.py#L61).

At the end of the *training.log* you will see the relevant metrics including the final F1 score a classification report. For further details on how to perform an evaluation for such a model, check [Notebook: Evaluating a Sequence Labeling Model](#).



```python
# INITIALIZE TRAINER
from flair.trainers import ModelTrainer

# define output path
model_folder = ner_model_path/'ner-model-test'

# option to continue from checkpoint
continue_training = False

if continue_training:
    checkpoint = tagger.load_checkpoint(model_folder/'checkpoint.pt')
    trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)
else:
    trainer = ModelTrainer(tagger, corpus)

# Training
trainer.train(model_folder,
              learning_rate=0.5,
              anneal_factor=0.5,
              mini_batch_size=8,
              patience=5,
              max_epochs=50,
              train_with_dev=True,
              monitor_test=True,
              shuffle=True,
              checkpoint=True)
```

---
## Variant 2 - K-Fold Cross-Validation
This section explains how to simulate 10-Fold Cross-Validation (CV) while training the ner model. CV is useful, when you want to reliably evaluate how a specific model configuration performs, when you do not have a specific test dataset.


### Sequence Labeling Dataset (Corpus)
Same as before.


```python
# PREPARE CORPUS
from flair.datasets import ColumnCorpus, SentenceDataset

# define columns (multiple possible: ... 2: 'pos')
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
corpus_folder = ner_corpus_path/'example-corpus'

# init a corpus using column format, data folder 
corpus = ColumnCorpus(corpus_folder, columns,
                      train_file='train.txt',
                      test_file='test.txt',
                      dev_file='dev.txt')
print(corpus)

# what tag do we want to predict?
tag_type = 'ner'

# make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
#print(tag_dictionary.idx2item)
```

### Embeddings
Same as before.


```python
# INITIALIZE EMBEDDINGS
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

# path to embeddings
lm_fwd = emb_path/'FLAIR'/'example-fwd'/'best-lm.pt'
lm_bwd = emb_path/'FLAIR'/'example-bwd'/'best-lm.pt'

embeddings = StackedEmbeddings([FlairEmbeddings(lm_fwd), FlairEmbeddings(lm_bwd)])
# alternative: embeddings = StackedEmbeddings([FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])
```

### Evaluation & Model Trainer
Since we use CV to evaluate model performance, we will first define a function that takes the *result* from  *model.evaluate()* and returns three pandas dataframes


*   ***tag_tfpn***: true-positive, false-postive, false-negative counts for each tag

*   ***tag_metrics***: values for precision, recall, accuracy and f1-scores for each tag
*   ***metrics***: values for precision, recall and f1-score

with the intention to sum or average those over each cross validation fold.



```python
import pandas as pd
import numpy as np
import re

def result_summary(result):
    scores = []
    lines = result.detailed_results.split('\n')
    for line in lines[3:]:
        split_line = re.split('\ -\ |\ +|:\ ', line)
        scores.append(split_line)    
    scores = np.array(scores)
    tags = scores[:,0].tolist()
    scores_ = scores[:, 2::2]
    tag_tfpn = scores_[:, :3].astype(int)
    tag_metrics = scores_[:, 4:].astype(float)
    metrics = np.array(result.log_line.split('\t')).astype(float).reshape(1,3)
  
    df_tag_tfpn = pd.DataFrame(data=tag_tfpn,index=tags,columns=['true-positive','false-positive', 'false-negative'])
    df_tag_metrics = pd.DataFrame(data=tag_metrics,index=tags,columns=['precision','recall', 'accuracy','f1-score'])
    df_metrics = pd.DataFrame(data=metrics, index=None,columns=['precision','recall','f1-score'])
  
    return df_tag_tfpn, df_tag_metrics, df_metrics
```

Next, we will perform the actual Cross-Validation: For every Fold, we will set *corpus.test* and *corpus.train* explicitly to subset of *complete_corpus*. In  this case, we leave *corpus.dev* empty since we are planning to ```train_with_dev``` anyway.

After each fold, we evaluate the model on the current test set. In the end, we get three dataframes:

*   ***details***: the sum of the true-false-positive-negative values for each tag (and resulting values for precision, recall and f1-score)
*   ***summary***: the average precision-recall-f1score
*   ***tag_metrics_avg***: the average precision-recall-accuracy-f1score values for each tag

These dataframes are stored as pickle files.





```python
from sklearn.model_selection import KFold
from flair.datasets import DataLoader, SentenceDataset
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


# Set number of splits
kf = KFold(n_splits=10)

# All sentences
complete_corpus = corpus.get_all_sentences()

# Cross-Validation
i=1
for train_index, test_index in kf.split(complete_corpus):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Fold:", i)
    corpus._train = SentenceDataset([complete_corpus[j] for j in train_index])
    corpus._test = SentenceDataset([complete_corpus[j] for j in test_index])
    corpus._dev = SentenceDataset([])
    print(corpus)
    
    # Initialize Sequence Tagger
    tagger = SequenceTagger(hidden_size=512,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type)
    
    # Initialize ModelTrainer
    trainer = ModelTrainer(tagger, corpus)
    # Define output path
    model_folder = ner_model_path/'ner-model-test-CV'
    
    # Training
    trainer.train(model_folder,
                  learning_rate=0.5,
                  anneal_factor=0.5,
                  mini_batch_size=8,
                  patience=5,
                  max_epochs=50,
                  train_with_dev=True,
                  monitor_test=True,
                  shuffle=False,
                  save_final_model=True,)
    
    # Evaluation
    result, eval_loss = trainer.model.evaluate(DataLoader(trainer.corpus.test,
                                                          batch_size=8,
                                                          num_workers=4))
    # tag_tfpn, tag_metrics, metrics
    if i==1:
        tt, tm, m = result_summary(result)
    else:
        tt_, tm_, m_ = result_summary(result)
        tt = tt.append(tt_)
        tm = tm.append(tm_)
        m = m.append(m_)
    
    i+=1  
    
df = tt.groupby(tt.index).sum()
tag_metrics_avg = tm.groupby(tm.index).mean()
summary = m.mean()
    
df['precision'] = df['true-positive'] / (df['true-positive'] + df['false-positive'])
df['recall'] = df['true-positive'] / (df['true-positive'] + df['false-negative'])
df['f1-score'] = 2*df['precision']*df['recall'] / (df['precision'] + df['recall'])
    
# pickle dump
import pickle
pickle.dump(df,(model_folder/'details.pkl').open(mode='wb'))
pickle.dump(tag_metrics_avg,(model_folder/'tag_metrics_avg.pkl').open(mode='wb'))
pickle.dump(summary,(model_folder/'summary.pkl').open(mode='wb'))


```