from typing import List
from nlp.embedding import vectorizer
from nlp.embedding.tfidf import Tfidf
from domain.value_objects.requirement import Requirement
from ml.classification_task.classification import Classification
from nlp.text_preprocessing.preprocessing import TextPreprocessing


class ClassificationApp:

    def run(self, requirements: List[Requirement]):
        classified_requirements = {}
        for requirement in requirements:
            text = requirement.__getattribute__("text")
            p_requirement = TextPreprocessing().run(text)
            v_requirement = vectorizer.run(Tfidf, p_requirement)
            label = Classification().classify(v_requirement)
            classified_requirements.update(
                {
                   "description": text,
                    "category": label,
                }
            )
        return classified_requirements


