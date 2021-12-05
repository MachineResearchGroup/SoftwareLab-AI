from dataclasses import dataclass
from application.nlp.embedding import vectorizer
from application.nlp.embedding.tfidf import Tfidf
from domain.value_objects.requirement import Requirement
# from application.ml.classification_task.classification import Classification
from application.nlp.embedding.embedding_interface import EmbeddingInterface
from application.nlp.text_preprocessing.preprocessing import TextPreprocessing


@dataclass
class ClassificationApp:

    # embedding: EmbeddingInterface

    def run(self, requirement: Requirement) -> dict:
        classified_requirements = {}
        text = requirement.__getattribute__("text")
        # txt_preprocessing = TextPreprocessing().run(text)
        # txt_vectorized = vectorizer.run(self.embedding, txt_preprocessing)
        # label = Classification().classify(v_requirement)
        classified_requirements.update(
            {
                "description": text,
                "category": "LF",
            }
        )
        return classified_requirements
