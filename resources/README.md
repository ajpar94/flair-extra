The resources folder can be used for storing trained models, evaluation results, plots, corpora, etc. A possible folder structure could look like this:

```
.
├── corpora
│   ├── classification_corpora
│   ├── column_corpora
│   └── text_corpora
│
└── models
    ├── classifiers
    ├── embeddings
    │   ├── ELMO
    │   ├── FASTTEXT
    │   ├── FLAIR
    │   └── WORD2VEC
    └── taggers
    
```

To create this exact structure, cd into resources/ and run:

```terminal
$ mkdir -p {corpora/{classification_corpora,column_corpora,text_corpora},models/{classifiers,taggers,embeddings/{ELMO,FASTTEXT,FLAIR,WORD2VEC}}}
```