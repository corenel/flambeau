from abc import abstractmethod


class BaseCriterion:
    @abstractmethod
    def compute(self, *inputs):
        pass
