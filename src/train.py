import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import extend_with_plan, load_calendar, load_plan
from src.features import make_training_dataset
from src.model import (
    BOOSTING_PARAMS,
    ForecastModel,
    fit_forecast_model,
    save_model,
)
from src.prophet_model import attach_prophet, prophet_forecasts
from src.validation import (
    baseline_predictions,
    cross_validation_cutoffs,
    interval_coverage,
    metrics_table,
    time_split,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / 'reports'
METHODS = [
    'baseline_week_ago',
    'baseline_two_weeks_ago',
    'prophet',
    'boosting',
]
ROW_COLUMNS = [
    'store_id',
    'date',
    'target_date',
    'horizon',
    'y',
    'target_dow',
    'target_promo',
]


def holdout_origins(
    holdouts: dict[int, pd.DataFrame],
) -> dict[int, list[pd.Timestamp]]:
    origins = pd.concat(
        [rows[['store_id', 'date']] for rows in holdouts.values()]
    ).drop_duplicates()
    return {
        int(store): sorted(group['date'])
        for store, group in origins.groupby('store_id')
    }


def evaluate_fold(
    frame: pd.DataFrame,
    training_sets: dict[int, pd.DataFrame],
    cutoff: pd.Timestamp,
    fold: int,
) -> pd.DataFrame:
    splits = {
        horizon: time_split(rows, cutoff)
        for horizon, rows in training_sets.items()
    }
    model = fit_forecast_model(
        {horizon: train for horizon, (train, _) in splits.items()}
    )
    forecasts = prophet_forecasts(
        frame,
        holdout_origins(
            {horizon: holdout for horizon, (_, holdout) in splits.items()}
        ),
    )
    parts = []
    for horizon, (_, holdout) in splits.items():
        rows = holdout.assign(horizon=horizon)
        parts.append(
            pd.concat(
                [
                    rows[ROW_COLUMNS],
                    baseline_predictions(holdout, horizon),
                    attach_prophet(holdout, forecasts, horizon).rename(
                        'prophet'
                    ),
                    model.predict(rows).rename(
                        columns={'forecast': 'boosting'}
                    ),
                ],
                axis=1,
            )
        )
    return pd.concat(parts, ignore_index=True).assign(fold=fold, cutoff=cutoff)


def run_validation(
    frame: pd.DataFrame,
    training_sets: dict[int, pd.DataFrame],
    cutoffs: list[pd.Timestamp],
) -> pd.DataFrame:
    return pd.concat(
        [
            evaluate_fold(frame, training_sets, cutoff, fold)
            for fold, cutoff in enumerate(cutoffs, start=1)
        ],
        ignore_index=True,
    )


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for (fold, horizon), group in predictions.groupby(['fold', 'horizon']):
        tables.append(
            metrics_table(group, group[METHODS], int(horizon)).assign(
                fold=fold
            )
        )
    for fold, group in predictions.groupby('fold'):
        tables.append(
            metrics_table(group, group[METHODS], 'все').assign(fold=fold)
        )
    return pd.concat(tables, ignore_index=True)


def coverage_table(predictions: pd.DataFrame) -> pd.DataFrame:
    records = [
        {
            'fold': fold,
            'store_id': str(store),
            'coverage, %': interval_coverage(
                group['y'], group['lower'], group['upper']
            ),
            'rows': len(group),
        }
        for (fold, store), group in predictions.groupby(['fold', 'store_id'])
    ]
    return pd.DataFrame(records)


def holdout_summary(metrics: pd.DataFrame, fold: int) -> dict[str, Any]:
    pooled = metrics[(metrics['fold'] == fold) & (metrics['horizon'] == 'все')]
    table = (
        pooled.set_index(['model', 'store_id'])[['MAE', 'MAPE, %']]
        .astype(float)
        .round(1)
    )
    return {
        str(model): group.droplevel('model').to_dict('index')
        for model, group in table.groupby(level='model')
    }


def cv_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    pooled = metrics[
        (metrics['horizon'] == 'все') & (metrics['store_id'] == 'все')
    ]
    return (
        pooled.groupby('model')['MAE']
        .agg(['mean', 'std', 'min', 'max'])
        .loc[METHODS]
        .round(1)
    )


def log_experiment(
    holdout: dict[str, Any], cv: pd.DataFrame, coverage: float
) -> Path:
    path = REPORTS_DIR / 'experiments.csv'
    record = pd.DataFrame(
        [
            {
                'run_at': datetime.now().isoformat(timespec='seconds'),
                'boosting_params': json.dumps(BOOSTING_PARAMS),
                **{
                    f'holdout_mae_{model}': holdout[model]['все']['MAE']
                    for model in METHODS
                },
                **{
                    f'cv_mae_{model}': cv.loc[model, 'mean']
                    for model in METHODS
                },
                'holdout_coverage, %': round(coverage, 1),
            }
        ]
    )
    record.to_csv(path, mode='a', header=not path.exists(), index=False)
    return path


def train() -> ForecastModel:
    frame = extend_with_plan(load_calendar(), load_plan())
    training_sets = make_training_dataset(frame)
    cutoffs = cross_validation_cutoffs(training_sets[1])
    holdout_fold = len(cutoffs)

    predictions = run_validation(frame, training_sets, cutoffs)
    metrics = fold_metrics(predictions)
    coverage = coverage_table(predictions)
    holdout = holdout_summary(metrics, holdout_fold)
    cv = cv_summary(metrics)
    holdout_coverage = interval_coverage(
        *(
            predictions.loc[predictions['fold'] == holdout_fold, column]
            for column in ('y', 'lower', 'upper')
        )
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    predictions.to_csv(
        REPORTS_DIR / 'predictions.csv', index=False, date_format='%Y-%m-%d'
    )
    metrics.to_csv(REPORTS_DIR / 'metrics.csv', index=False)
    coverage.to_csv(REPORTS_DIR / 'coverage.csv', index=False)
    log_experiment(holdout, cv, holdout_coverage)

    model = fit_forecast_model(training_sets)
    model.metadata.update(
        {
            'holdout_start': str((cutoffs[-1] + pd.Timedelta(days=1)).date()),
            'holdout_end': str(predictions['target_date'].max().date()),
            'holdout_metrics': holdout,
            'holdout_coverage': round(holdout_coverage, 1),
        }
    )
    save_model(model)
    print('Holdout (MAE, MAPE %, все горизонты):')
    print(json.dumps(holdout, ensure_ascii=False, indent=2))
    print('Бэктест, MAE по окнам:')
    print(cv.to_string())
    print(f'Покрытие интервала на holdout: {holdout_coverage:.1f}%')
    return model


if __name__ == '__main__':
    train()
