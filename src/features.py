import pandas as pd

HORIZONS = tuple(range(1, 8))
HISTORY_LAGS = tuple(range(7))
ROLLING_WINDOWS = (7, 28)
SAME_DOW_WEEKS = (1, 2, 3, 4)
SAME_DOW_EXTRA_WEEKS = (2, 3, 4)
LAST_OPEN_LIMIT = 7

HISTORY_FEATURES = [
    *[f'lag_{k}' for k in HISTORY_LAGS],
    *[f'rolling_mean_{w}' for w in ROLLING_WINDOWS],
    'rolling_std_28',
    'last_open',
]
SAME_DOW_FEATURES = [
    *[f'same_dow_{w}w_ago' for w in SAME_DOW_EXTRA_WEEKS],
    'same_dow_mean',
]
TARGET_DAY_FEATURES = [
    'target_dow',
    'target_month',
    'target_is_december',
    'target_is_short_day',
    'target_promo',
    'target_school_holiday',
    'target_is_pre_holiday',
]

CATEGORICAL_FEATURES = ['store_id', 'target_dow', 'target_month']
FEATURE_COLUMNS = [
    'store_id',
    *HISTORY_FEATURES,
    *SAME_DOW_FEATURES,
    *TARGET_DAY_FEATURES,
]


def check_calendar(df: pd.DataFrame) -> None:
    steps = df.groupby('store_id')['date'].diff().dropna()
    if not steps.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            'Календарь с дырами в датах: сначала вызовите restore_calendar'
        )


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    target = df.groupby('store_id')['target']
    features = {f'lag_{k}': target.shift(k) for k in HISTORY_LAGS}
    for window in ROLLING_WINDOWS:
        rolling = target.rolling(window, min_periods=window // 2)
        features[f'rolling_mean_{window}'] = rolling.mean().reset_index(
            level=0, drop=True
        )
    features['rolling_std_28'] = (
        target.rolling(28, min_periods=14)
        .std()
        .reset_index(level=0, drop=True)
    )
    features['last_open'] = target.ffill(limit=LAST_OPEN_LIMIT)
    return pd.DataFrame(features, index=df.index)


def same_dow_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = df.groupby('store_id')['target']
    weeks = pd.DataFrame(
        {
            f'same_dow_{w}w_ago': target.shift(7 * w - horizon)
            for w in SAME_DOW_WEEKS
        },
        index=df.index,
    )
    return weeks[[f'same_dow_{w}w_ago' for w in SAME_DOW_EXTRA_WEEKS]].assign(
        same_dow_mean=weeks.mean(axis=1)
    )


def day_features(df: pd.DataFrame) -> pd.DataFrame:
    date = df['date']
    is_holiday = df['state_holiday'].isin(['a', 'b', 'c']).astype(int)
    return pd.DataFrame(
        {
            'dow': date.dt.dayofweek + 1,
            'month': date.dt.month,
            'is_december': (date.dt.month == 12).astype(int),
            'is_short_day': (
                (date.dt.month == 12) & date.dt.day.isin([24, 31])
            ).astype(int),
            'promo': df['promo'],
            'school_holiday': df['school_holiday'],
            'is_pre_holiday': is_holiday.groupby(df['store_id']).shift(-1),
        },
        index=df.index,
    )


def make_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError(
            f'Горизонт должен быть от {HORIZONS[0]} до {HORIZONS[-1]}, получено {horizon}'
        )
    df = df.sort_values(['store_id', 'date']).reset_index(drop=True)
    check_calendar(df)

    target_day = (
        day_features(df)
        .groupby(df['store_id'])
        .shift(-horizon)
        .add_prefix('target_')
    )
    y = df.groupby('store_id')['target'].shift(-horizon)

    return pd.concat(
        [
            df[['store_id', 'date']].assign(
                target_date=df['date'] + pd.Timedelta(days=horizon)
            ),
            history_features(df),
            same_dow_features(df, horizon),
            target_day[TARGET_DAY_FEATURES],
            y.rename('y'),
        ],
        axis=1,
    )


HISTORY_CHECK_FEATURE = 'rolling_mean_28'
INTEGER_FEATURES = {'target_dow': 'int64', 'target_month': 'int64'}


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    has_answer = df['y'].notna()
    has_history = df[HISTORY_CHECK_FEATURE].notna()
    return df[has_history & has_answer].astype(INTEGER_FEATURES)


def split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df['y']


def make_all_horizons(calendar: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {horizon: make_features(calendar, horizon) for horizon in HORIZONS}


def make_training_dataset(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        horizon: training_rows(frame)
        for horizon, frame in make_all_horizons(df).items()
    }


def forecast_rows(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        features = make_features(frame, horizon)
        rows.append(
            features[features['date'] == origin].assign(horizon=horizon)
        )
    result = pd.concat(rows, ignore_index=True)
    if result.empty or result[HISTORY_CHECK_FEATURE].isna().any():
        raise ValueError(
            f'Слишком короткая история до {origin.date()}: '
            'нужно хотя бы 14 открытых дней за последние 28'
        )
    return result.drop(columns='y').astype(INTEGER_FEATURES)
