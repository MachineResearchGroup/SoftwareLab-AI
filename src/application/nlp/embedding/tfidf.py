from embedding_interface import EmbeddingInterface
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass


@dataclass
class Tfidf(EmbeddingInterface):
    """"""

    vectorizer: TfidfVectorizer = TfidfVectorizer()

    def fit(self, tokens: list):
        self.vectorizer.fit(tokens)

    def vectorize(self, tokens: list):
        return self.vectorizer.fit_transform(tokens)

    def key_words(self):
        return self.vectorizer.get_feature_names_out()
