from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / 'data' / 'raw' / 'train.csv'
RAW_PLAN_PATH = ROOT / 'data' / 'raw' / 'test.csv'
PROCESSED_PATH = ROOT / 'data' / 'processed' / 'calendar.csv'
PLAN_PATH = ROOT / 'data' / 'processed' / 'plan.csv'

COLUMNS = {
    'Store': 'store_id',
    'DayOfWeek': 'day_of_week',
    'Date': 'date',
    'Sales': 'sales',
    'Customers': 'customers',
    'Open': 'open',
    'Promo': 'promo',
    'StateHoliday': 'state_holiday',
    'SchoolHoliday': 'school_holiday',
}

SCHEMA = {
    'store_id': 'int64',
    'day_of_week': 'int64',
    'sales': 'int64',
    'customers': 'int64',
    'open': 'int64',
    'promo': 'int64',
    'state_holiday': 'str',
    'school_holiday': 'int64',
}

PLAN_SOURCE_COLUMNS = (
    'Store',
    'DayOfWeek',
    'Date',
    'Open',
    'Promo',
    'StateHoliday',
    'SchoolHoliday',
)
PLAN_COLUMNS = {name: COLUMNS[name] for name in PLAN_SOURCE_COLUMNS}
PLAN_SCHEMA = {
    column: dtype
    for column, dtype in SCHEMA.items()
    if column in PLAN_COLUMNS.values()
}
PLAN_FIELDS = [
    'store_id',
    'date',
    'promo',
    'state_holiday',
    'school_holiday',
    'status',
]

BINARY_COLUMNS = ('open', 'promo', 'school_holiday')
STATE_HOLIDAYS = ('0', 'a', 'b', 'c')

STORE_IDS = (1, 20)
FORECAST_PLAN_DAYS = 8

STATUS_CLOSE = 0
STATUS_OPEN = 1
NO_DATA_STATUS = -1


def read_table(
    path: Path,
    columns: dict[str, str],
    schema: dict[str, str],
    store_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Нет файла {path}. Загрузить - make download')
    df = pd.read_csv(path, dtype={'StateHoliday': str})
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f'Данные не совпадают со схемой. Не хватает колонок: {missing}'
        )
    df = df[list(columns)].rename(columns=columns)

    if store_ids is not None:
        unknown = set(store_ids) - set(df['store_id'].to_list())
        if unknown:
            raise ValueError(f'В датасете отсутствуют магазины: {unknown}')
        df = df[df['store_id'].isin(store_ids)]

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise ValueError(
            f'В данных есть пропуски (колонка, количество): {nulls.to_dict()}'
        )

    df = df.assign(
        date=pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    )
    bad_dates = int(df['date'].isna().sum())
    if bad_dates:
        raise ValueError(
            f'Неверный формат дат ({bad_dates} ошибок). Ожидается Year-Month-Day'
        )
    return df.astype(schema)


def common_checks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        'наличие дубликатов дат': df.duplicated(['store_id', 'date']),
        'значения не 0/1 в open, promo, school_holiday': ~df[
            list(BINARY_COLUMNS)
        ]
        .isin([0, 1])
        .all(axis=1),
        'неизвестные коды state_holiday': ~df['state_holiday'].isin(
            STATE_HOLIDAYS
        ),
        'day_of_week не совпадает с датой': df['day_of_week']
        != df['date'].dt.dayofweek + 1,
    }


def raise_on_errors(checks: dict[str, pd.Series]) -> None:
    errors = {
        check: int(mask.sum()) for check, mask in checks.items() if mask.any()
    }
    if errors:
        raise ValueError(
            f'В данных есть ошибки (тип, количество строк):\n{errors}'
        )


def validate_raw(df: pd.DataFrame) -> None:
    raise_on_errors(
        {
            **common_checks(df),
            'отрицательные продажи или покупатели': (
                df[['sales', 'customers']] < 0
            ).any(axis=1),
            'закрыт, но есть покупатели': (df['open'] == 0)
            & (df['customers'] > 0),
        }
    )


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop(columns='day_of_week')
        .sort_values(['store_id', 'date'])
        .reset_index(drop=True)
    )


def load_raw(
    path: Path = RAW_PATH, store_ids: Sequence[int] | None = None
) -> pd.DataFrame:
    df = read_table(path, COLUMNS, SCHEMA, store_ids)
    validate_raw(df)
    return finalize(df)


def restore_calendar(
    raw_df: pd.DataFrame, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    frames = []
    for store_id, store_df in raw_df.groupby('store_id'):
        first = store_df['date'].min()
        last = (
            store_df['date'].max()
            if end is None
            else max(end, store_df['date'].max())
        )
        days = pd.date_range(first, last, freq='D', name='date')
        frames.append(
            store_df.set_index('date')
            .reindex(days)
            .assign(store_id=store_id)
            .reset_index()
        )
    return pd.concat(frames, ignore_index=True)[raw_df.columns]


def open_to_status(open_flag: pd.Series) -> np.ndarray:
    return np.select(
        [open_flag == 0, open_flag == 1],
        [STATUS_CLOSE, STATUS_OPEN],
        default=NO_DATA_STATUS,
    )


def add_status_and_target(df: pd.DataFrame) -> pd.DataFrame:
    status = open_to_status(df['open'])
    return df.assign(
        status=status, target=df['customers'].where(status == STATUS_OPEN)
    ).drop(columns=['customers', 'open', 'sales'])


def prepare_calendar(
    path: Path = RAW_PATH, store_ids: Sequence[int] | None = STORE_IDS
) -> pd.DataFrame:
    return add_status_and_target(restore_calendar(load_raw(path, store_ids)))


def prepare_plan(
    path: Path = RAW_PLAN_PATH, store_ids: Sequence[int] | None = STORE_IDS
) -> pd.DataFrame:
    df = read_table(path, PLAN_COLUMNS, PLAN_SCHEMA, store_ids)
    raise_on_errors(common_checks(df))
    df = finalize(df)
    return df.assign(status=open_to_status(df['open']))[PLAN_FIELDS]


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format='%Y-%m-%d')
    return path


def read_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f'Нет файла {path}. Подготовить - make calendar'
        )
    return pd.read_csv(
        path, parse_dates=['date'], dtype={'state_holiday': str}
    )


def load_calendar(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    return read_processed(path)


def load_plan(path: Path = PLAN_PATH) -> pd.DataFrame:
    return read_processed(path)


def known_plan(
    calendar: pd.DataFrame, plan: pd.DataFrame, store_id: int
) -> pd.DataFrame:
    history_plan = calendar.loc[
        (calendar['store_id'] == store_id)
        & (calendar['status'] != NO_DATA_STATUS),
        PLAN_FIELDS,
    ]
    future_plan = plan.loc[plan['store_id'] == store_id, PLAN_FIELDS]
    return (
        pd.concat([history_plan, future_plan], ignore_index=True)
        .drop_duplicates('date', keep='first')
        .set_index('date')
    )


def extend_with_plan(
    calendar: pd.DataFrame, plan: pd.DataFrame
) -> pd.DataFrame:
    last_dates = calendar.groupby('store_id')['date'].max()
    after_history = plan['date'] > plan['store_id'].map(last_dates)
    future = plan[after_history].assign(target=np.nan)
    return (
        pd.concat([calendar, future[calendar.columns]], ignore_index=True)
        .sort_values(['store_id', 'date'])
        .reset_index(drop=True)
    )


def build_forecast_frame(
    calendar: pd.DataFrame,
    plan: pd.DataFrame,
    store_id: int,
    origin: pd.Timestamp,
    plan_days: int = FORECAST_PLAN_DAYS,
) -> pd.DataFrame:
    store_calendar = calendar[calendar['store_id'] == store_id]
    if store_calendar.empty:
        known = sorted(calendar['store_id'].unique().tolist())
        raise ValueError(
            f'Магазина {store_id} нет в данных. Доступные магазины: {known}'
        )
    first, last = store_calendar['date'].min(), store_calendar['date'].max()
    if origin < first:
        raise ValueError(
            f'История магазина {store_id} начинается {first.date()}: '
            f'прогноз на дату раньше {(first + pd.Timedelta(days=1)).date()} '
            'построить нельзя'
        )
    if origin > last:
        raise ValueError(
            f'История магазина {store_id} заканчивается {last.date()}: '
            'самая поздняя дата начала прогноза - '
            f'{(last + pd.Timedelta(days=1)).date()}'
        )

    future_dates = pd.date_range(
        origin + pd.Timedelta(days=1), periods=plan_days, freq='D', name='date'
    )
    future = known_plan(calendar, plan, store_id).reindex(future_dates)
    missing = future.index[future['status'].isna()]
    if not missing.empty:
        dates = ', '.join(str(day.date()) for day in missing)
        raise ValueError(
            f'Нет плана работы, промо и праздников на даты: {dates}. '
            'Прогноз строится только на даты, для которых есть план сети'
        )

    history = store_calendar[store_calendar['date'] <= origin]
    future = future.reset_index().assign(target=np.nan)
    return pd.concat([history, future[history.columns]], ignore_index=True)


if __name__ == '__main__':
    print(
        f'Календарь сохранен: {save_table(prepare_calendar(), PROCESSED_PATH)}'
    )
    print(f'План сети сохранен: {save_table(prepare_plan(), PLAN_PATH)}')
