import pandas as pd
import pytest

from src.features import HORIZONS, make_training_dataset
from src.validation import (
    HOLDOUT_WEEKS,
    baseline_predictions,
    cross_validation_cutoffs,
    interval_coverage,
    metrics_table,
    split_cutoff,
    time_split,
)


@pytest.fixture
def training_sets(calendar: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return make_training_dataset(calendar)


def test_train_answers_are_known_before_first_holdout_forecast(
    training_sets: dict[int, pd.DataFrame],
) -> None:
    cutoff = split_cutoff(training_sets[1])
    for rows in training_sets.values():
        train, holdout = time_split(rows, cutoff)
        assert train['target_date'].max() <= holdout['date'].min()
        assert holdout['target_date'].min() > cutoff


def test_holdout_covers_same_days_for_all_horizons(
    training_sets: dict[int, pd.DataFrame],
) -> None:
    cutoff = split_cutoff(training_sets[1])
    days = {
        horizon: set(time_split(rows, cutoff)[1]['target_date'])
        for horizon, rows in training_sets.items()
    }
    assert all(days[horizon] == days[1] for horizon in HORIZONS)
    assert max(days[1]) - cutoff <= pd.Timedelta(weeks=HOLDOUT_WEEKS)


def test_time_split_rejects_cutoff_without_history(
    training_sets: dict[int, pd.DataFrame],
) -> None:
    rows = training_sets[1]
    with pytest.raises(ValueError, match='Недостаточно истории'):
        time_split(rows, rows['date'].min() - pd.Timedelta(days=1))


def test_cross_validation_cutoffs_step_back_by_holdout(
    training_sets: dict[int, pd.DataFrame],
) -> None:
    cutoffs = cross_validation_cutoffs(training_sets[1], folds=3)
    steps = pd.Series(cutoffs).diff().dropna()
    assert steps.eq(pd.Timedelta(weeks=HOLDOUT_WEEKS)).all()
    assert cutoffs[-1] == split_cutoff(training_sets[1])


@pytest.mark.parametrize('horizon', HORIZONS)
def test_baselines_use_same_weekday_in_the_past(
    calendar: pd.DataFrame,
    training_sets: dict[int, pd.DataFrame],
    horizon: int,
) -> None:
    rows = training_sets[horizon]
    target = calendar.set_index(['store_id', 'date'])['target']
    baselines = baseline_predictions(rows, horizon)
    for weeks, column in [
        (1, 'baseline_week_ago'),
        (2, 'baseline_two_weeks_ago'),
    ]:
        keys = list(
            zip(
                rows['store_id'],
                rows['target_date'] - pd.Timedelta(weeks=weeks),
                strict=True,
            )
        )
        expected = target.reindex(keys).to_numpy()
        known = ~pd.isna(expected)
        assert (baselines[column].to_numpy()[known] == expected[known]).all()


def test_metrics_on_simple_values() -> None:
    holdout = pd.DataFrame({'store_id': [1, 2], 'y': [100.0, 200.0]})
    predictions = pd.DataFrame({'model': [110.0, 180.0]})
    table = metrics_table(holdout, predictions, 1).set_index('store_id')

    assert table.loc['все', 'MAE'] == pytest.approx(15.0)
    assert table.loc['все', 'MAPE, %'] == pytest.approx(10.0)
    assert table.loc['1', 'MAPE, %'] == pytest.approx(10.0)
    assert table.loc['все', 'rows'] == 2
    assert interval_coverage(
        holdout['y'], pd.Series([90.0, 210.0]), pd.Series([120.0, 220.0])
    ) == pytest.approx(50.0)
