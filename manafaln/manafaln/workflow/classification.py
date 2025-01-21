from manafaln.workflow import SupervisedLearning

class SupervisedClassification(SupervisedLearning):
    """
    A class representing a supervised classification model.

    This class inherits from the SupervisedLearning class and provides additional functionality specific to classification tasks.

    Parameters:
    - config (dict): A dictionary containing the configuration parameters for the model.

    Attributes:
    - config (dict): A dictionary containing the configuration parameters for the model.

    """

    def __init__(self, config: dict):
        """
        Initializes a new instance of the SupervisedClassification class.

        Parameters:
        - config (dict): A dictionary containing the configuration parameters for the model.

        """
        super().__init__(config)

