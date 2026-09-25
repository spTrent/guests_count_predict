from pathlib import Path

import pandas as pd
import pytest

from src.data import build_forecast_frame
from src.features import forecast_rows, make_training_dataset
from src.model import (
    ForecastModel,
    fit_forecast_model,
    load_model,
    save_model,
)
from src.prophet_model import attach_prophet, holidays_table


@pytest.fixture(scope='module')
def model() -> ForecastModel:
    from tests.conftest import make_calendar

    return fit_forecast_model(make_training_dataset(make_calendar()))


def test_forecast_is_non_negative_and_inside_interval(
    model: ForecastModel, calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    origin = calendar['date'].max()
    rows = forecast_rows(
        build_forecast_frame(calendar, plan, 1, origin), origin
    )
    forecast = model.predict(rows)

    assert list(forecast.index) == list(rows.index)
    assert (forecast['forecast'] >= 0).all()
    assert (forecast['lower'] <= forecast['forecast']).all()
    assert (forecast['forecast'] <= forecast['upper']).all()


def test_model_survives_save_and_load(
    model: ForecastModel,
    calendar: pd.DataFrame,
    plan: pd.DataFrame,
    tmp_path: Path,
) -> None:
    origin = calendar['date'].max()
    rows = forecast_rows(
        build_forecast_frame(calendar, plan, 2, origin), origin
    )
    loaded = load_model(save_model(model, tmp_path / 'model.joblib'))

    pd.testing.assert_frame_equal(model.predict(rows), loaded.predict(rows))
    assert loaded.metadata['stores'] == [1, 2]


def test_load_model_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='make train'):
        load_model(tmp_path / 'missing.joblib')


def test_predict_rejects_unknown_horizon(
    model: ForecastModel, calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    origin = calendar['date'].max()
    rows = forecast_rows(
        build_forecast_frame(calendar, plan, 1, origin), origin
    )
    with pytest.raises(ValueError, match='Нет модели'):
        model.predict(rows.assign(horizon=9))


def test_prophet_holidays_and_matching(calendar: pd.DataFrame) -> None:
    holidays = holidays_table(calendar)
    assert holidays['holiday'].tolist() == ['easter']
    assert holidays[['lower_window', 'upper_window']].iloc[0].tolist() == [
        -2,
        1,
    ]

    rows = pd.DataFrame(
        {
            'store_id': [1, 2],
            'date': pd.to_datetime(['2024-03-01', '2024-03-01']),
        }
    )
    forecasts = pd.DataFrame(
        {
            'store_id': [1, 1],
            'date': pd.to_datetime(['2024-03-01', '2024-03-01']),
            'horizon': [1, 2],
            'prophet': [150.0, 160.0],
        }
    )
    matched = attach_prophet(rows, forecasts, horizon=2)
    assert matched.iloc[0] == 160.0
    assert pd.isna(matched.iloc[1])
