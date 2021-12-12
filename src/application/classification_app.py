from dataclasses import dataclass
import logging
from infra.local_data import (get_dataset)
from application.nlp.embedding import vectorizer
from domain.value_objects.requirement import Requirement
from application.ml.classification_task.classification import Classification
from application.nlp.embedding.embedding_interface import EmbeddingInterface
from application.nlp.text_preprocessing.preprocessing import TextPreprocessing
from domain.value_objects.classifiers.classifier_interface import ClassifierInterface


@dataclass
class ClassificationApp:

    embedding: EmbeddingInterface
    classification: Classification

    def run(self, requirement: Requirement) -> dict:
        classified_requirements = {}
        text = requirement.__getattribute__("text")
        txt_preprocessing = " ".join(TextPreprocessing().run(text))
        logging.info(f"Text preprocessing: {txt_preprocessing}")
        txt_vectorized = vectorizer.run(self.embedding, [txt_preprocessing])
        logging.info(f"Text vectorized: {txt_vectorized}")
        label = self.classification.classify(txt_vectorized)
        classified_requirements.update(
            {
                "description": text,
                "category": label[0],
            }
        )
        return classified_requirements

    def fit_model(self):
        dataset = get_dataset()
        data = dataset["RequirementText"]
        x = []
        y = dataset["Class"]
        for text in data:
            x.append(
                " ".join(TextPreprocessing().run(text))
            )
        x = vectorizer.run(self.embedding, x)
        self.classification.training(x, y)
