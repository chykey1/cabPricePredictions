"""Module to support the creation of model objects."""
import datetime
from typing import NamedTuple, Union


class PipelineDateParams(NamedTuple):
    """Named tuple containing date parameters for the CrossValidationPipeline
    class.

    Args:
        start_date (Union[str, datetime.datetime]): Date time at which to start
            cross validation.
        period (str): Periods for which to do rolling cross validations.
        folds (int): Number of folds over which to cross validate.
    """

    start_date: Union[str, datetime.datetime]
    period: str
    folds: int


class CrossValidationPipeline:
    def __init__(self, date_params: PipelineDateParams):
        """Performs rolling cross validation over dates with a given custom
        model class and hyperparameters.

        Args:
            date_params (PipelineDateParams): Parameter tuple containing
                parameters to use when cross validating.
        """
        self.date_parameters = self.calculate_date_folds(date_params)

    @staticmethod
    def calculate_date_folds(date_params: PipelineDateParams) -> dict:
        """Calculates the date folds based on

        Args:
            date_params (PipelineDateParams): Parameters to use to calculate
                dates required.

        Returns:
            Dictionary containing train_end and test_end for each fold.
        """
        pass
