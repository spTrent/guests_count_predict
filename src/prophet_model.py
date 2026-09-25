import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.features import HORIZONS

RANDOM_STATE = 42
REGRESSORS = ('promo', 'school_holiday')
HOLIDAY_WINDOWS = {
    'a': ('public_holiday', -1, 0),
    'b': ('easter', -2, 1),
    'c': ('christmas', -14, 1),
}


def holidays_table(frame: pd.DataFrame) -> pd.DataFrame:
    days = frame.loc[
        frame['state_holiday'].isin(list(HOLIDAY_WINDOWS)),
        ['date', 'state_holiday'],
    ].drop_duplicates()
    windows = days['state_holiday'].map(HOLIDAY_WINDOWS)
    return pd.DataFrame(
        {
            'ds': days['date'],
            'holiday': windows.str[0],
            'lower_window': windows.str[1],
            'upper_window': windows.str[2],
        }
    ).reset_index(drop=True)


def fit_prophet(history: pd.DataFrame, holidays: pd.DataFrame) -> Any:
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').disabled = True
    from prophet import Prophet

    model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    for regressor in REGRESSORS:
        model.add_regressor(regressor)
    train = history.rename(columns={'date': 'ds', 'target': 'y'})
    return model.fit(train[['ds', 'y', *REGRESSORS]], seed=RANDOM_STATE)


def store_forecasts(
    store_frame: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    holidays = holidays_table(store_frame)
    steps = pd.to_timedelta(list(horizons), unit='D')
    parts = []
    for origin in origins:
        history = store_frame[
            (store_frame['date'] <= origin) & store_frame['target'].notna()
        ]
        future = store_frame[
            store_frame['date'].isin(origin + steps)
            & store_frame[list(REGRESSORS)].notna().all(axis=1)
        ]
        if history.empty or future.empty:
            continue
        model = fit_prophet(history, holidays)
        predicted = model.predict(
            future.rename(columns={'date': 'ds'})[['ds', *REGRESSORS]]
        )
        parts.append(
            pd.DataFrame(
                {
                    'date': origin,
                    'target_date': future['date'].to_numpy(),
                    'prophet': predicted['yhat'].clip(lower=0).to_numpy(),
                }
            )
        )
    result = pd.concat(parts, ignore_index=True)
    return result.assign(
        horizon=(result['target_date'] - result['date']).dt.days
    )


def prophet_forecasts(
    frame: pd.DataFrame, origins: dict[int, list[pd.Timestamp]]
) -> pd.DataFrame:
    parts = [
        store_forecasts(
            frame[frame['store_id'] == store_id].reset_index(drop=True),
            store_origins,
        ).assign(store_id=store_id)
        for store_id, store_origins in origins.items()
    ]
    return pd.concat(parts, ignore_index=True)


def attach_prophet(
    rows: pd.DataFrame, forecasts: pd.DataFrame, horizon: int
) -> pd.Series:
    keys = ['store_id', 'date']
    matched = rows[keys].merge(
        forecasts.loc[forecasts['horizon'] == horizon, [*keys, 'prophet']],
        on=keys,
        how='left',
    )
    return pd.Series(matched['prophet'].to_numpy(), index=rows.index)
