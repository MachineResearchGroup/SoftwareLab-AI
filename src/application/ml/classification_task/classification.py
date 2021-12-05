import logging
import sklearn
from typing import List
from dataclasses import dataclass
from infra.local_data import export_sklearn_model
from application.ml.classification_task import optimization
from domain.value_objects.classifiers.classifier_interface import ClassifierInterface


@dataclass
class Classification:

    inter: int
    k_folds: int
    model: sklearn
    classifier: ClassifierInterface

    def classify(self, x: str) -> str:
        return self.model.predict(x)

    def training(self, x, y):
        logging.info("\nTraining the algorithm...")
        optimized_model = optimization.get_optimized_model(
            self.classifier, x, y)
        export_sklearn_model(model=optimized_model,
                             file_name="lr_model.joblib")
