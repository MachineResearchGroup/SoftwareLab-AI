from typing import List
from domain.value_objects.requirement import Requirement


class Mapper:

    def to_requirement(data: str) -> List[Requirement]:
        """"""
        return Requirement(data, None)
