from dataclasses import dataclass
from skopt.space import Real, Categorical, Integer


@dataclass
class ClassifierInterface:

    name: str
    initials: str
    model: object
    instance: object
    search_spaces: dict
