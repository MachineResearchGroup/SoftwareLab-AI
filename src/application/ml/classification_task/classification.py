import logging
from typing import List
import infra.local_data as local_data
from dataclasses import dataclass
from application.ml.classification_task import optimization
from domain.value_objects.classifiers.classifier_interface import ClassifierInterface


@dataclass
class Classification:

    inter: int
    k_folds: int
    classifier: local_data.get_sklearn_model("sklearn_model.joblib")

    def classify(self, x: str) -> str:
        return self.classifier.predict(x)

    def training(self, x, y: List[str], classifier: ClassifierInterface):
        logging.info("\nTraining the algorithm...")
        model = optimization.get_optimized_model(classifier, x, y)
        local_data.export_sklearn_model(model=model)
