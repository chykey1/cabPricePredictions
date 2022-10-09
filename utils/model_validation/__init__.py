"""Module to support the creation of model objects."""
import datetime
from typing import NamedTuple

from dateutil.relativedelta import relativedelta


class PipelineDateParams(NamedTuple):
    """Named tuple containing date parameters for the CrossValidationPipeline
    class.

    Args:
        start_date (Union[str, datetime.datetime]): Date time at which to start
            cross validation.
        period (str): Periods for which to do rolling cross validations.
        folds (int): Number of folds over which to cross validate.
    """

    start_date: datetime.datetime
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

        assert type(date_params.start_date) == datetime.datetime

        date_folds = dict()

        for fold_number in range(0, date_params.folds):
            train_end = date_params.start_date + relativedelta(
                **{date_params.period: fold_number}
            )
            test_end = train_end + relativedelta(**{date_params.period: 1})

            date_folds[f"fold_{fold_number + 1}"] = {
                "train_end": train_end,
                "test_end": test_end,
            }

        return date_folds
