import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

from tqdm import tqdm
from typing import List, Union
from flair.datasets import ColumnCorpus, ClassificationCorpus, CSVClassificationCorpus
from flair.data import Corpus, Sentence
from flair.visual.ner_html import render_ner_html
from IPython.core.display import display, HTML
from wordcloud import WordCloud
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

plt.rc('axes', axisbelow=True)


class CorpusAnalysis():
    def __init__(
        self,
        path: Union[Path, str],
        corpus: Corpus,
    ):
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()

        self.path = path
        self.corpus = corpus
        self.sentences = self.corpus.get_all_sentences()
        print(self.corpus)

    def obtain_statistics(
        self,
        tag_type: str = 'ner',
        save_as_json: bool = True
    ):
        stats_splits = self.corpus.obtain_statistics(tag_type)
        stats_complete = json.dumps(Corpus._obtain_statistics_for(self.sentences, 'complete', tag_type), indent=4)
        if save_as_json:
            (self.path / 'stats_splits.json').write_text(stats_splits)
            (self.path / 'stats_complete.json').write_text(stats_complete)

        return (stats_splits, stats_complete)

    def most_common_tokens(
        self,
        nr_tokens: int = 15,
        min_freq: int = 5,
        stop_words=[]
    ):
        # get most common tokens
        top = self.corpus._get_most_common_tokens(max_tokens=-1, min_freq=min_freq)
        top_filtered = [t for t in top if t.lower() not in stop_words and len(t) > 1][:nr_tokens]

        return top_filtered

    def wordcloud(
        self,
        figsize=((10, 10)),
        savefig_file=None,
        **kwargs
    ):
        text = (' ').join([s.to_plain_string() for s in self.sentences])
        # Create and generate a word cloud image:
        wordcloud = WordCloud(**kwargs).generate(text)
        # Display the generated image:
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        if savefig_file:
            plt.savefig(self.path / savefig_file, dpi=300)
        plt.show()


class ColumnCorpusAnalysis(CorpusAnalysis):
    def __init__(
        self,
        path: Union[Path, str],
        columns: dict = None,
        tag_types: List = ['ner'],
        corpus: Corpus = None
    ):
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()

        self.path = path
        if corpus:
            self.corpus = corpus
        else:
            self.corpus = ColumnCorpus(self.path, columns)
        self.sentences = ColumnCorpusAnalysis.index_spans(self.corpus.get_all_sentences(), tag_types)
        print(self.corpus)

    def visualize_ner_tags(
        self,
        display_index=range(5),
        save_display_html: bool = False,
        save_all_html: bool = True,
        **kwargs
    ):
        html = render_ner_html([self.sentences[i] for i in display_index], **kwargs)
        display(HTML(html))
        if save_display_html:
            (self.path / 'sentences_true_example.html').write_text(html)
        if save_all_html:
            html = render_ner_html(self.sentences, **kwargs)
            (self.path / 'sentences_true_all.html').write_text(html)

    def tag_distribution(
        self,
        nr_tags: int = 10,
        tag_type: str = 'ner',
        savefig_file=None,
        **kwargs
    ):
        support = defaultdict(int)
        for sentence in self.sentences:
            true_tags = [tag.tag for tag in sentence.get_spans(tag_type)]
            for tag in true_tags:
                support[tag] += 1
        support = pd.DataFrame.from_dict(support, orient='index', columns=['support']).sort_values('support', ascending=False)
        html_table = support.to_html()

        # (self.path/'support.html').write_text(html_table)

        support_top = support[:nr_tags].copy()
        if nr_tags < len(support):
            support_top.loc['others'] = support[nr_tags:].sum()
        # pie plot support
        support_top.plot.pie(y='support', **kwargs)
        plt.legend(labels=support_top.index, bbox_to_anchor=(1, 0, 0.1, 1), loc='center right')
        plt.tight_layout()
        if savefig_file:
            plt.savefig(self.path / savefig_file, dpi=600)
        plt.show()

    def most_common_tokens_per_tags(
        self,
        tag_type: str = 'ner',
        max_tokens: int = None,
        convert_to_lower: bool = True,
        print_without_count: bool = False
    ):
        mc_tokens = {}
        for sentence in self.sentences:
            tag_tuples = [(tag.tag, tag.text) for tag in sentence.get_spans(tag_type)]
            for tag, text in tag_tuples:
                if tag not in mc_tokens.keys():
                    mc_tokens[tag] = defaultdict(int)
                if convert_to_lower:
                    mc_tokens[tag][text.lower()] += 1
                else:
                    mc_tokens[tag][text] += 1

        for key, value in mc_tokens.items():
            mc_tokens[key] = Counter(value).most_common(max_tokens)

        # nice printing
        for key, value in mc_tokens.items():
            if print_without_count:
                value = [tokens for (tokens, counts) in value]
            print('{:<20}{}'.format(str(key), str(value)))

    @staticmethod
    def index_spans(sentences, tag_types):
        untagged = [Sentence(s.to_original_text()) for s in sentences]
        for i in range(len(untagged)):
            tokens = sentences[i].tokens
            for j, token in enumerate(tokens):
                for tag_type in tag_types:
                    label = token.get_tag(tag_type)
                    untagged[i].tokens[j].add_tag_label(tag_type, label)
        return untagged


class ClassificationCorpusAnalysis(CorpusAnalysis):
    def __init__(
        self,
        path: Union[Path, str],
        column_name_map: dict = None,
        corpus: Corpus = None,
        **corpus_params
    ):
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()

        self.path = path
        if corpus:
            self.corpus = corpus
        else:
            if column_name_map:
                self.corpus = CSVClassificationCorpus(self.path, column_name_map, **corpus_params)
            else:
                self.corpus = ClassificationCorpus(self.path, **corpus_params)
        self.sentences = self.corpus.get_all_sentences()
        print(self.corpus)

    def class_distribution(
        self,
        multiclass: bool = False,
        nr_classes: int = 10,
        savefig_file=None,
        **kwargs
    ):
        class_count = Corpus._get_class_to_count(self.sentences)
        class_count = pd.DataFrame.from_dict(class_count, orient='index', columns=['count']).sort_values('count', ascending=False)
        html_table = class_count.to_html()

        # plot distribution
        class_count_top = class_count[:nr_classes].copy()
        if not multiclass:
            if nr_classes < len(class_count):
                class_count_top.loc['others'] = class_count[nr_classes:].sum()
            # pie plot class_count
            class_count_top.plot.pie(y='count', **kwargs)
            plt.legend(labels=class_count_top.index, bbox_to_anchor=(1, 0, 0.1, 1), loc='center right')
        else:
            class_count_top.plot.bar(y='count', **kwargs)
            plt.gca().yaxis.grid(True, linestyle='--')

        plt.tight_layout()
        if savefig_file:
            plt.savefig(self.path / savefig_file, dpi=600)
        plt.show()

    def example_document_for_classes(
        self,
    ):
        # Todo!
        pass


class TextAnalysis():
    def __init__(
        self,
        path: Union[Path, str]
    ):
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()

        self.path = path
        self.token_counter = defaultdict(int)
        tokens_in_line = []
        with self.path.open(mode='r') as textfile:
            for i, line in enumerate(tqdm(textfile)):
                tokens = line.split()
                tokens_in_line.append(len(tokens))
                for token in tokens:
                    self.token_counter[token] += 1

        self.token_counter = Counter(self.token_counter)
        self.statistics = {
            "file_name": self.path.name,
            "total_number_of_lines": len(tokens_in_line),
            "number_of_tokens": {
                "total": sum(tokens_in_line),
                "max": max(tokens_in_line),
                "min": min(tokens_in_line),
                "avg": sum(tokens_in_line) / len(tokens_in_line)
            }
        }

    def obtain_statistics(
        self,
        save_as_json: bool = True
    ):
        stats = json.dumps(self.statistics, indent=4)
        if save_as_json:
            (self.path.parent / 'stats.json').write_text(stats)

        return stats

    def most_common_tokens(
        self,
        nr_tokens: int = 15,
        min_freq: int = 5,
        stop_words=[],
        without_count: bool = True
    ):
        # get most common tokens
        if stop_words:
            mct = TextAnalysis._filter_token_counter(self.token_counter, stop_words).most_common(nr_tokens)
        else:
            mct = self.token_counter.most_common(nr_tokens)
        if without_count:
            mct = [token for (token, count) in mct]

        return mct

    def wordcloud(
        self,
        figsize=((10, 10)),
        stop_words=[],
        savefig_file=None,
        **kwargs
    ):
        if stop_words:
            counter = TextAnalysis._filter_token_counter(self.token_counter, stop_words)
        else:
            counter = self.token_counter

        # Create and generate a word cloud image:
        wordcloud = WordCloud(**kwargs).generate_from_frequencies(counter)
        # Display the generated image:
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        if savefig_file:
            plt.savefig(self.path.parent / savefig_file, dpi=600)
        plt.show()

    @staticmethod
    def _filter_token_counter(
        counter: Counter,
        stop_words: list
    ):
        counter_filter = counter.copy()
        for t, f in counter.items():
            if t.lower() in stop_words or len(t) <= 1:
                del counter_filter[t]

        return counter_filter
