from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'train.csv'
PROCESSED_PATH = (
    Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'calendar.csv'
)

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

BINARY_COLUMNS = ('open', 'promo', 'school_holiday')
STATE_HOLIDAYS = ('0', 'a', 'b', 'c')

STORE_IDS = (1, 20)


def load_raw(
    path: Path = RAW_PATH, store_ids: Sequence[int] | None = None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Нет файла {path}. Загрузить - make download')
    df = pd.read_csv(path, dtype={'StateHoliday': str})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f'Данные не совпадают с схемой. Не хватает колонок: {missing}'
        )
    df = df[list(COLUMNS)].rename(columns=COLUMNS)

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

    df = df.astype(SCHEMA)
    validate_raw(df)
    return (
        df.drop(columns='day_of_week')
        .sort_values(['store_id', 'date'])
        .reset_index(drop=True)
    )


def validate_raw(df: pd.DataFrame) -> None:
    checks = {
        'наличие дубликатов дат': df.duplicated(['store_id', 'date']),
        'отрицательные продажи или покупатели': (
            df[['sales', 'customers']] < 0
        ).any(axis=1),
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
        'закрыт, но есть покупатели': (df['open'] == 0)
        & (df['customers'] > 0),
    }
    errors = {
        check: int(mask.sum()) for check, mask in checks.items() if mask.any()
    }
    if errors:
        raise ValueError(
            f'В данных есть ошибки (тип, количество строк):\n{errors}'
        )


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


STATUS_CLOSE = 0
STATUS_OPEN = 1
NO_DATA_STATUS = -1


def add_status_and_target(df: pd.DataFrame) -> pd.DataFrame:
    status = np.select(
        [df['open'] == 0, df['open'] == 1],
        [STATUS_CLOSE, STATUS_OPEN],
        default=NO_DATA_STATUS,
    )
    return df.assign(
        status=status, target=df['customers'].where(df['open'] == STATUS_OPEN)
    ).drop(columns=['customers', 'open', 'sales'])


def prepare_calendar(
    path: Path = RAW_PATH, store_ids: Sequence[int] | None = STORE_IDS
) -> pd.DataFrame:
    return add_status_and_target(restore_calendar(load_raw(path, store_ids)))


def save_calendar(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format='%Y-%m-%d')
    return path


def load_calendar(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Нет файла {path}. Загрузить - make calendar')
    return pd.read_csv(
        path, parse_dates=['date'], dtype={'state_holiday': str}
    )


if __name__ == '__main__':
    print(
        f'Датасет с восстановленными датами сохранен: {save_calendar(prepare_calendar())}'
    )
