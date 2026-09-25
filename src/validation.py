import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

HOLDOUT_WEEKS = 6
CV_FOLDS = 4


def split_cutoff(
    df: pd.DataFrame, holdout_weeks: int = HOLDOUT_WEEKS
) -> pd.Timestamp:
    return df['target_date'].max() - pd.Timedelta(weeks=holdout_weeks)


def time_split(
    df: pd.DataFrame, cutoff: pd.Timestamp, holdout_weeks: int = HOLDOUT_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_end = cutoff + pd.Timedelta(weeks=holdout_weeks)
    holdout = df[
        (df['target_date'] > cutoff) & (df['target_date'] <= holdout_end)
    ]
    if holdout.empty:
        raise ValueError(f'Нет данных для проверки после {cutoff.date()}')
    train = df[df['target_date'] <= holdout['date'].min()]
    if train.empty:
        raise ValueError(
            f'Недостаточно истории для разбиения по дате {cutoff.date()}'
        )
    return train, holdout


def cross_validation_cutoffs(
    rows: pd.DataFrame,
    folds: int = CV_FOLDS,
    holdout_weeks: int = HOLDOUT_WEEKS,
) -> list[pd.Timestamp]:
    last = rows['target_date'].max()
    return [
        last - pd.Timedelta(weeks=holdout_weeks * fold)
        for fold in range(folds, 0, -1)
    ]


def baseline_predictions(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    fallback = df['same_dow_mean']
    return pd.DataFrame(
        {
            'baseline_week_ago': df[f'lag_{7 - horizon}'].fillna(fallback),
            'baseline_two_weeks_ago': df['same_dow_2w_ago'].fillna(fallback),
        },
        index=df.index,
    )


def metrics_table(
    holdout: pd.DataFrame, predictions: pd.DataFrame, horizon: int | str
) -> pd.DataFrame:
    valid = predictions.notna().all(axis=1)
    frame = predictions[valid].assign(
        y=holdout.loc[valid, 'y'], store_id=holdout.loc[valid, 'store_id']
    )
    groups = [
        ('все', frame),
        *((str(store), group) for store, group in frame.groupby('store_id')),
    ]
    records = [
        {
            'horizon': horizon,
            'store_id': store,
            'model': name,
            'MAE': mean_absolute_error(group['y'], group[name]),
            'MAPE, %': mean_absolute_percentage_error(group['y'], group[name])
            * 100,
            'rows': len(group),
        }
        for store, group in groups
        for name in predictions.columns
    ]
    return pd.DataFrame(records)


def interval_coverage(
    y_true: pd.Series, lower: pd.Series, upper: pd.Series
) -> float:
    return float(((y_true >= lower) & (y_true <= upper)).mean() * 100)
