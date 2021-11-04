import logging
from embedding_interface import EmbeddingInterface


def run(approach: EmbeddingInterface, tokens: list):
    """approach -> TF-IDF or Word2Vec"""

    logging.info('\nVectoring the requirements text...')
    return approach.vectorize(tokens)
