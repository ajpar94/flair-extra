import pickle
from pathlib import Path
from flair.models import SequenceTagger
from flair.datasets import DataLoader
from flair.data import Sentence
from segtok.segmenter import split_single
from flair.visual.ner_html import render_ner_html
from flair.visual.training_curves import Plotter
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
from typing import List, Union
import spacy
from nltk.stem.snowball import GermanStemmer

# !python -m spacy download de_core_news_sm
plt.rc('axes', axisbelow=True)


class SequenceTaggerEvaluation():
    def __init__(
        self,
        path: Union[Path, str],
        model: str = 'final-model.pt'
    ):
        if type(path) == str:
            path = Path(path)
        assert path.exists()

        self.path = path
        self.model = SequenceTagger.load(path / model)
        self.cv_results = {}
        for file in ['summary', 'details']:
            try:
                self.cv_results[file] = pickle.load((path / (file + '.pkl')).open(mode='rb'))
            except FileNotFoundError:
                print(f"{file+'.pkl'} not found. Setting cv_results['{file}'] to None")

        self.plotter = Plotter()

    def result_tables(
        self,
        save_as_html: bool = True
    ):
        html_0 = self.cv_results['summary'].to_frame('value').to_html()
        html_1 = self.cv_results['details'].to_html()
        display(HTML(html_0))
        print('\n')
        display(HTML(html_1))

        if save_as_html:
            (self.path / 'summary.html').write_text(html_0)
            (self.path / 'details.html').write_text(html_1)

    def plot_tag_stats(
        self,
        mode: str,
        savefig: bool = False,
        **kwargs
    ):
        """
        mode
        tp-fn: stacked barplot - true-positives and false-negatives
        tp-fp: bar plot - true-positives and false-positives
        """
        details = self.cv_results['details']

        if mode == 'tp_fn':
            details[['true-positive', 'false-negative']].plot.bar(stacked=True, **kwargs)
        elif mode == 'tp_fp':
            details[['true-positive', 'false-positive']].plot.bar(stacked=False, **kwargs)
        else:
            details[mode.split('_')].plot.bar(stacked=False, **kwargs)

        plt.gca().yaxis.grid(True, linestyle='--')
        plt.tight_layout()
        if savefig:
            plt.savefig(self.path / (mode + '.png'))

    def confusion_matrix(
        self,
    ):
        # confusion matrix tags
        pass

    def predict(
        self,
        sentences: Union[str, Sentence, List[Sentence], List[str]],
        display_html: bool = True,
        html_file: str = None,
        display_str: bool = False,
        **kwargs
    ):
        if type(sentences) == Sentence:
            sentences = [sentences]
        elif type(sentences) == str:
            sentences = split_single(sentences)

        if type(sentences[0]) == str:
            sentences = [Sentence(s, use_tokenizer=True) for s in sentences]

        self.model.predict(sentences)

        if display_html or html_file:
            html = render_ner_html(sentences, **kwargs)
            if display_html:
                display(HTML(html))
            if html_file:
                (self.path / html_file).write_text(html)
        if display_str:
            for sentence in sentences:
                print(sentence.to_tagged_string())

    def plot_training_curves(
        self,
        plot_values: List[str] = ["loss", "F1"]
    ):
        self.plotter.plot_training_curves(self.path / 'loss.tsv', plot_values)

    def plot_weights(self):
        self.plotter.plot_weights(self.path / 'weights.txt')

    def plot_learning_rate(
        self,
        skip_first: int = 10,
        skip_last: int = 5
    ):
        self.plotter.plot_learning_rate(self.path / 'loss.tsv', skip_first, skip_last)

    @staticmethod
    def _preprocess(text, mode=None):
        '''helper function to preprocess text. returns List of Sentences'''
        sentences = split_single(text)
        if mode:
            nlp = spacy.load('de_core_news_sm')
            if mode == 'lemmatize':
                sentences = [Sentence((' ').join([token.lemma_ for token in nlp(s)])) for s in sentences]
            elif mode == 'stem':
                stemmer = GermanStemmer()
                sentences = [Sentence((' ').join([stemmer.stem(token.text) for token in nlp(s)])) for s in sentences]
        else:
            sentences = [Sentence(s, use_tokenizer=True) for s in sentences]

        return sentences
