from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    NO_DATA_STATUS,
    STATUS_CLOSE,
    STATUS_OPEN,
    add_status_and_target,
    build_forecast_frame,
    extend_with_plan,
    load_raw,
    restore_calendar,
)

RAW_HEADER = 'Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday'
RAW_ROWS = [
    '1,1,2024-01-01,1000,100,1,1,0,0',
    '1,2,2024-01-02,0,0,0,0,a,0',
    '1,4,2024-01-04,1200,120,1,1,0,1',
]


def write_raw(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / 'train.csv'
    path.write_text('\n'.join([RAW_HEADER, *rows]) + '\n')
    return path


def test_restore_calendar_separates_three_kinds_of_days(
    tmp_path: Path,
) -> None:
    raw = load_raw(write_raw(tmp_path, RAW_ROWS), store_ids=[1])
    calendar = add_status_and_target(restore_calendar(raw)).set_index('date')

    assert calendar.loc['2024-01-01', 'status'] == STATUS_OPEN
    assert calendar.loc['2024-01-01', 'target'] == 100
    assert calendar.loc['2024-01-02', 'status'] == STATUS_CLOSE
    assert pd.isna(calendar.loc['2024-01-02', 'target'])
    assert calendar.loc['2024-01-03', 'status'] == NO_DATA_STATUS
    assert pd.isna(calendar.loc['2024-01-03', 'target'])
    assert {'customers', 'sales', 'open'}.isdisjoint(calendar.columns)


@pytest.mark.parametrize(
    ('rows', 'message'),
    [
        ([*RAW_ROWS, RAW_ROWS[0]], 'дубликатов'),
        (['1,2,2024-01-02,0,15,0,0,a,0'], 'закрыт, но есть покупатели'),
        (['1,3,2024-01-02,1000,100,1,0,0,0'], 'day_of_week'),
        (['1,2,2024-01-02,1000,100,1,0,x,0'], 'state_holiday'),
        (['1,2,02.01.2024,1000,100,1,0,0,0'], 'формат дат'),
    ],
)
def test_load_raw_rejects_bad_data(
    tmp_path: Path, rows: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_raw(write_raw(tmp_path, rows))


def test_load_raw_rejects_unknown_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='отсутствуют магазины'):
        load_raw(write_raw(tmp_path, RAW_ROWS), store_ids=[1, 99])


def test_load_raw_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='make download'):
        load_raw(tmp_path / 'missing.csv')


def test_extend_with_plan_appends_future_without_answers(
    calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    extended = extend_with_plan(calendar, plan)
    future = extended['date'] > calendar['date'].max()

    assert len(extended) == len(calendar) + len(plan)
    assert extended.loc[future, 'target'].isna().all()
    assert (
        extended.groupby('store_id')['date']
        .diff()
        .dropna()
        .eq(pd.Timedelta(days=1))
        .all()
    )


def test_build_forecast_frame_rejects_unknown_store(
    calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match='Магазина 99 нет'):
        build_forecast_frame(calendar, plan, 99, calendar['date'].max())


def test_build_forecast_frame_rejects_date_after_history(
    calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    origin = calendar['date'].max() + pd.Timedelta(days=3)
    with pytest.raises(ValueError, match='заканчивается'):
        build_forecast_frame(calendar, plan, 1, origin)


def test_build_forecast_frame_requires_plan_for_all_days(
    calendar: pd.DataFrame, plan: pd.DataFrame
) -> None:
    short_plan = plan[
        plan['date'] <= plan['date'].min() + pd.Timedelta(days=3)
    ]
    with pytest.raises(ValueError, match='Нет плана'):
        build_forecast_frame(calendar, short_plan, 1, calendar['date'].max())
