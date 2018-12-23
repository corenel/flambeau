from abc import abstractmethod


class BaseCriterion:
    @abstractmethod
    def compute_loss(self, *inputs):
        pass
