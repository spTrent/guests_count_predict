import numpy as np
import pandas as pd
import pytest

START = pd.Timestamp('2024-01-01')
DAYS = 160
HOLIDAY = pd.Timestamp('2024-03-29')
GAP_STORE = 2
GAP_START = 40
GAP_DAYS = 10
PLAN_DAYS = 14


def store_calendar(store_id: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    dow = dates.dayofweek.to_numpy()
    week = ((dates - START).days // 7).to_numpy()
    is_holiday = dates == HOLIDAY
    status = np.where((dow == 6) | is_holiday, 0, 1)
    target = np.where(
        status == 1,
        100.0 * store_id + 10 * dow + np.arange(len(dates)) % 5,
        np.nan,
    )
    return pd.DataFrame(
        {
            'store_id': store_id,
            'date': dates,
            'promo': ((week % 2 == 0) & (dow < 5)).astype(float),
            'state_holiday': np.where(is_holiday, 'b', '0'),
            'school_holiday': (week % 5 == 0).astype(float),
            'status': status,
            'target': target,
        }
    )


def make_calendar(store_ids: tuple[int, ...] = (1, GAP_STORE)) -> pd.DataFrame:
    dates = pd.date_range(START, periods=DAYS, freq='D')
    calendar = pd.concat(
        [store_calendar(store_id, dates) for store_id in store_ids],
        ignore_index=True,
    )
    gap_dates = dates[GAP_START : GAP_START + GAP_DAYS]
    gap = (calendar['store_id'] == GAP_STORE) & calendar['date'].isin(
        gap_dates
    )
    calendar.loc[
        gap, ['promo', 'state_holiday', 'school_holiday', 'target']
    ] = np.nan
    calendar.loc[gap, 'status'] = -1
    return calendar


def make_plan(calendar: pd.DataFrame, days: int = PLAN_DAYS) -> pd.DataFrame:
    last = calendar['date'].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=days, freq='D')
    plan = pd.concat(
        [
            store_calendar(int(store_id), dates)
            for store_id in calendar['store_id'].unique()
        ],
        ignore_index=True,
    )
    return plan.drop(columns='target').astype(
        {'promo': int, 'school_holiday': int}
    )


@pytest.fixture
def calendar() -> pd.DataFrame:
    return make_calendar()


@pytest.fixture
def plan(calendar: pd.DataFrame) -> pd.DataFrame:
    return make_plan(calendar)
