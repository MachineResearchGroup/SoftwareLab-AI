from infra.local_data import get_dataset, export_sklearn_model
from application.nlp.embedding.tfidf import Tfidf


def fit_tfidf():
    corpus = get_dataset()
    tfidf_model = Tfidf().fit(corpus)
    export_sklearn_model(tfidf_model, "/models/tfidf_model.joblib")