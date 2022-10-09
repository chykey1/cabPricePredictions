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
