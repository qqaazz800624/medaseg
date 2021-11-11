from manafaln.workflow import SupervisedLearning

class SupervisedClassification(SupervisedLearning):
    def __init__(self, config: dict):
        super().__init__(config)
