from application.classification_app import ClassificationApp
from application.nlp.embedding.tfidf import Tfidf
from infra.local_data import (has_file)


class DependencyInjector:

    tfidf = Tfidf() if not has_file("models/tfidf_model.joblib") else ""
    classification_app = ClassificationApp()
