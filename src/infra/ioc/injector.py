from application.nlp.embedding.tfidf import Tfidf
from application.embedding_app import EmbeddingApp
from infra.local_data import (has_file, get_sklearn_model)
from application.classification_app import ClassificationApp
from application.ml.classification_task.classification import Classification
from domain.value_objects.classifiers.logistic_regression import LogisticRegressionClass


class DependencyInjector:

    if not has_file("models/tfidf_model.joblib"):
        EmbeddingApp.fit_tfidf()

    tfidf_model = get_sklearn_model("models/tfidf_model.joblib")
    tfidf = Tfidf(vectorizer=tfidf_model)

    classifier = LogisticRegressionClass()

    classification = Classification(
        inter=150,
        k_folds=5,
        model=None,
        classifier=classifier
    )
    classification_app = ClassificationApp(
        embedding=tfidf, classification=classification)

    if not has_file("models/sklearn_model.joblib"):
        classification_app.fit_model()
    sklearn_model = get_sklearn_model("sklearn_model.joblib")
    classification.__setattr__("model", sklearn_model)