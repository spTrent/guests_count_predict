import numpy as np
import pandas as pd
import pytest

from src.data import build_forecast_frame, extend_with_plan
from src.features import (
    FEATURE_COLUMNS,
    HISTORY_LAGS,
    HORIZONS,
    forecast_rows,
    make_features,
    make_training_dataset,
)

CUTOFF = pd.Timestamp('2024-04-15')


@pytest.mark.parametrize('horizon', HORIZONS)
def test_features_do_not_look_into_future(
    calendar: pd.DataFrame, horizon: int
) -> None:
    changed = calendar.copy()
    future = changed['date'] > CUTOFF
    changed.loc[future, 'target'] = changed.loc[future, 'target'] * 10 + 999

    original = make_features(calendar, horizon)
    modified = make_features(changed, horizon)
    past = original['date'] <= CUTOFF

    pd.testing.assert_frame_equal(
        original.loc[past, FEATURE_COLUMNS],
        modified.loc[past, FEATURE_COLUMNS],
    )


@pytest.mark.parametrize('horizon', HORIZONS)
def test_lags_and_answer_point_to_right_days(
    calendar: pd.DataFrame, horizon: int
) -> None:
    features = make_features(calendar, horizon)
    target = calendar.set_index(['store_id', 'date'])['target']
    row = features[
        (features['store_id'] == 1) & (features['date'] == CUTOFF)
    ].iloc[0]

    for lag in HISTORY_LAGS:
        expected = target.get((1, CUTOFF - pd.Timedelta(days=lag)), np.nan)
        assert row[f'lag_{lag}'] == pytest.approx(expected, nan_ok=True)

    target_day = CUTOFF + pd.Timedelta(days=horizon)
    two_weeks_ago = target_day - pd.Timedelta(days=14)
    assert row['target_date'] == target_day
    assert row['y'] == pytest.approx(target[(1, target_day)], nan_ok=True)
    assert row['same_dow_2w_ago'] == pytest.approx(
        target[(1, two_weeks_ago)], nan_ok=True
    )
    assert row['target_dow'] == target_day.dayofweek + 1


@pytest.mark.parametrize('horizon', [0, 8, -1])
def test_rejects_unknown_horizon(calendar: pd.DataFrame, horizon: int) -> None:
    with pytest.raises(ValueError, match='Горизонт'):
        make_features(calendar, horizon)


def test_rejects_calendar_with_missing_dates(calendar: pd.DataFrame) -> None:
    broken = calendar.drop(index=calendar.index[5])
    with pytest.raises(ValueError, match='restore_calendar'):
        make_features(broken, 1)


def test_training_rows_have_answer_and_history(calendar: pd.DataFrame) -> None:
    for rows in make_training_dataset(calendar).values():
        assert rows['y'].notna().all()
        assert rows['rolling_mean_28'].notna().all()
        assert rows['target_dow'].between(1, 6).all()


@pytest.mark.parametrize('store_id', [1, 2])
def test_forecast_rows_match_training_features(
    calendar: pd.DataFrame, plan: pd.DataFrame, store_id: int
) -> None:
    extended = extend_with_plan(calendar, plan)
    origins = [CUTOFF, calendar['date'].max()]
    for origin in origins:
        frame = build_forecast_frame(calendar, plan, store_id, origin)
        rows = forecast_rows(frame, origin).set_index('horizon')
        for horizon in HORIZONS:
            full = make_features(extended, horizon)
            expected = full[
                (full['store_id'] == store_id) & (full['date'] == origin)
            ][FEATURE_COLUMNS].iloc[0]
            np.testing.assert_allclose(
                rows.loc[horizon, FEATURE_COLUMNS].astype(float),
                expected.astype(float),
                equal_nan=True,
            )


def test_forecast_rows_hide_future_answers(
    calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    frame = build_forecast_frame(calendar, plan, 1, CUTOFF)
    assert frame.loc[frame['date'] > CUTOFF, 'target'].isna().all()


def test_forecast_rows_reject_short_history(
    calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    origin = calendar['date'].min() + pd.Timedelta(days=5)
    frame = build_forecast_frame(calendar, plan, 1, origin)
    with pytest.raises(ValueError, match='короткая история'):
        forecast_rows(frame, origin)
