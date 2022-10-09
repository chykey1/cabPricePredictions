"""Module to support the creation of model objects."""
import datetime
from typing import NamedTuple

import pandas as pd
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

    def run_pipeline(
        self,
        data_frame: pd.DataFrame,
        model_instance: callable,
        model_parameters: NamedTuple,
    ) -> dict:
        """For each fold in the pipelines `date_parameters` method will run
        `model_instance` with `model_parameters` on the train and test splits.

        Args:
            data_frame (pd.DataFrame): pd.DataFrame to run cross validation
                pipeline over.
            model_instance (callable): Callable model object with fit and
                predict methods.
            model_parameters (NamedTuple): Named tuple containing model
                instances hyperparameters.
        Returns:
            A dictionary containing train and test predictions for each fold.
        """

        results = dict()

        for fold in self.date_parameters.keys():
            train = data_frame[
                data_frame.index < self.date_parameters[fold]["train_end"]
            ]
            test = data_frame[
                (data_frame.index >= self.date_parameters[fold]["train_end"])
                & (data_frame.index < self.date_parameters[fold]["test_end"])
            ]

            model = model_instance(model_parameters)
            model = model.train(train)

            test_predictions = model.predict(test)

            results[fold] = {
                "test_mae": test_predictions,
            }

        return results
