from pathlib import Path

import pandas as pd
import pytest

import predict
from src.data import save_table
from src.features import make_training_dataset
from src.model import (
    ForecastModel,
    fit_forecast_model,
    forecast_week,
    save_model,
)
from tests.conftest import make_calendar, make_plan


@pytest.fixture(scope='module')
def model() -> ForecastModel:
    return fit_forecast_model(make_training_dataset(make_calendar()))


@pytest.fixture
def files(tmp_path: Path, model: ForecastModel) -> dict[str, Path]:
    calendar = make_calendar()
    return {
        'model': save_model(model, tmp_path / 'model.joblib'),
        'calendar': save_table(calendar, tmp_path / 'calendar.csv'),
        'plan': save_table(make_plan(calendar), tmp_path / 'plan.csv'),
    }


def run(files: dict[str, Path], *args: str) -> int:
    paths = [
        item
        for name, path in files.items()
        for item in (f'--{name}', str(path))
    ]
    return predict.main([*args, *paths])


def test_forecast_week_closes_sundays_and_shows_actuals(
    model: ForecastModel, calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    start = pd.Timestamp('2024-04-15')
    forecast = forecast_week(model, calendar, plan, 1, start)

    assert forecast['date'].tolist() == list(
        pd.date_range(start, periods=7, freq='D')
    )
    sundays = forecast['date'].dt.dayofweek == 6
    assert forecast['closed'].equals(sundays)
    assert (
        (forecast.loc[sundays, ['forecast', 'lower', 'upper']] == 0)
        .all()
        .all()
    )
    assert forecast.loc[~sundays, 'actual'].notna().all()


def test_forecast_week_rejects_store_outside_model(
    model: ForecastModel, calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match='обучена только на магазинах'):
        forecast_week(model, calendar, plan, 3, calendar['date'].max())


def test_cli_prints_forecast_and_saves_csv(
    files: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / 'forecast.csv'
    start = make_calendar()['date'].max() + pd.Timedelta(days=1)

    code = run(
        files,
        '--date',
        f'{start:%Y-%m-%d}',
        '--restaurant',
        '1',
        '--output',
        str(output),
    )

    printed = capsys.readouterr().out
    assert code == 0
    assert 'Магазин 1' in printed
    assert 'закрыт по графику' in printed
    assert len(pd.read_csv(output)) == 7


@pytest.mark.parametrize(
    ('date', 'store', 'message'),
    [
        ('2024-06-20', '9', 'обучена только на магазинах'),
        ('2025-01-01', '1', 'заканчивается'),
        ('2024-01-03', '1', 'короткая история'),
    ],
)
def test_cli_reports_clear_errors(
    files: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    date: str,
    store: str,
    message: str,
) -> None:
    code = run(files, '--date', date, '--store', store)
    assert code == 1
    assert message in capsys.readouterr().err


def test_cli_rejects_bad_date_format(files: dict[str, Path]) -> None:
    with pytest.raises(SystemExit) as error:
        run(files, '--date', '15.08.2024', '--store', '1')
    assert error.value.code == 2
