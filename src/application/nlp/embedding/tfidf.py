from application.nlp.embedding.embedding_interface import EmbeddingInterface
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass


@dataclass
class Tfidf(EmbeddingInterface):
    """"""

    vectorizer: TfidfVectorizer

    def fit(self, tokens: list):
        self.vectorizer().fit(tokens)
        return self.vectorizer

    def vectorize(self, tokens: list):
        return self.vectorizer.transform(tokens)

    def key_words(self):
        return self.vectorizer.get_feature_names_out()
