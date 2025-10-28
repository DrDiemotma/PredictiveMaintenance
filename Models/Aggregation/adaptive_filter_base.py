from abc import ABC, abstractmethod
from collections.abc import Iterable


class FilterBase(ABC):
    @abstractmethod
    def next(self, value: float):
        pass

    def run(self, values: Iterable[float]) -> list:
        """
        Run the evaluation on a sequence of added data entries.
        :param values: The sequence of values the filter processes.
        :return: list of evaluation results.
        """
        return [self.next(x) for x in values]

