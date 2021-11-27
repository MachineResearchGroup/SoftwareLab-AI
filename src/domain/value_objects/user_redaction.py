from dataclasses import dataclass
from typing import List


@dataclass
class Redaction:
    """"""

    text: str
    sentences: List[str]
    key_words: List[str]
