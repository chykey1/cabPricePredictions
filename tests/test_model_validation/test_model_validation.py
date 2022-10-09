import datetime

import pytest

from utils.model_validation import CrossValidationPipeline, PipelineDateParams


@pytest.fixture()
def test_validation_pipeline():
    """Used to yield an instance of CrossValidationPipeline for testing.

    Returns:
        An instance of CrossValidationPipeline.
    """
    yield CrossValidationPipeline


@pytest.mark.parametrize("time_period", ["hours"])
def test_calculate_date_params(
    test_validation_pipeline: CrossValidationPipeline,
    time_period: str,
):
    """Tests calculate_date_folds method within CrossValidationPipeline class.

    Args:
        test_validation_pipeline (CrossValidationPipeline): Instance of
            CrossValidationPipeline to test with.
        time_period (str): Period to pass to PipelineDateParams.
    """
    fold_params = PipelineDateParams(
        start_date=datetime.datetime(2022, 1, 1, 8, 0, 0),
        period=time_period,
        folds=10,
    )

    test_validation_pipeline.calculate_date_folds(date_params=fold_params)

    date_folds = {
        "fold_1": {
            "train_end": datetime.datetime(2022, 1, 1, 8, 0, 0),
            "test_end": datetime.datetime(2022, 1, 1, 9, 0, 0),
        },
        "fold_2": {
            "train_end": datetime.datetime(2022, 1, 1, 9, 0, 0),
            "test_end": datetime.datetime(2022, 1, 1, 10, 0, 0),
        },
    }

    for fold in date_folds.values():
        assert isinstance(fold, dict)
        assert isinstance(fold["train_end"], datetime.datetime)
        assert isinstance(fold["test_end"], datetime.datetime)
        assert fold["train_end"] <= fold["test_end"]
        assert fold["test_end"] - fold["train_end"] == datetime.timedelta(
            **{time_period: 1}
        )
