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
        """Rolling validation over dates for a given custom model class and
            hyperparameters.

        Args:
            date_params (PipelineDateParams): Parameter tuple containing
                parameters to use when cross validating.
        """
        self.date_parameters = self.calculate_date_folds(date_params)
        self.results = dict()

    @staticmethod
    def calculate_date_folds(date_params: PipelineDateParams) -> dict:
        """Calculates the date folds based on the pipelines

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

    def fit_evaluate(
        self,
        data_frame: pd.DataFrame,
        model_instance: callable,
        model_parameters: NamedTuple,
    ) -> pd.DataFrame:
        """Running `model_instance` on the train and test splits for each fold
            in the pipelines `date_parameters` method with `model_parameters`.

        Args:
            data_frame (pd.DataFrame): pd.DataFrame to run cross validation
                pipeline over.
            model_instance (callable): Callable model object with fit and
                predict methods.
            model_parameters (NamedTuple): Named tuple containing model
                instances hyperparameters.
        Returns:
            A pd.DataFrame containing train and test predictions for each fold.
        """

        for fold in self.date_parameters.keys():
            train_frame = data_frame[
                data_frame.index < self.date_parameters[fold]["train_end"]
            ]
            test_frame = data_frame[
                (data_frame.index >= self.date_parameters[fold]["train_end"])
                & (data_frame.index < self.date_parameters[fold]["test_end"])
            ]

            model = model_instance(model_parameters)
            model = model.train(train_frame)

            test_predictions = model.predict(test_frame)

            self.results[fold] = {
                "test_predictions": test_predictions,
            }

        return self.evaluate_results(data_frame=data_frame)

    def evaluate_results(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Appending dictionary of predictions for each fold to test dataset.

        Args:
            data_frame (pd.DataFrame): Dataframe containing.
        Returns:
            Dataframe containing full test set across folds with predictions
                appended.
        """
        data_frame = data_frame.copy()

        full_predictions = pd.concat(self.results)
        test_start = self.date_parameters.values()[0]["test_start"]

        data_frame[data_frame.index >= test_start] = full_predictions

        return data_frame
