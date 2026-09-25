import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.data import STATUS_CLOSE, build_forecast_frame
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    forecast_rows,
    split_x_y,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'

RANDOM_STATE = 42
LOWER_QUANTILE = 0.1
UPPER_QUANTILE = 0.9
BOOSTING_PARAMS: dict[str, Any] = {
    'learning_rate': 0.05,
    'max_iter': 300,
    'max_leaf_nodes': 15,
    'min_samples_leaf': 20,
    'l2_regularization': 1.0,
}


def build_boosting(
    loss: str = 'absolute_error', quantile: float | None = None
) -> Pipeline:
    select = ColumnTransformer(
        [('features', 'passthrough', FEATURE_COLUMNS)],
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')
    model = HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        categorical_features=CATEGORICAL_FEATURES,
        random_state=RANDOM_STATE,
        **BOOSTING_PARAMS,
    )
    return Pipeline([('select', select), ('model', model)])


@dataclass
class ForecastModel:
    point: dict[int, Pipeline]
    lower: dict[int, Pipeline]
    upper: dict[int, Pipeline]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizons(self) -> list[int]:
        return sorted(self.point)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        unknown = set(rows['horizon']) - set(self.point)
        if unknown:
            raise ValueError(f'Нет модели для горизонтов: {sorted(unknown)}')
        parts = []
        for horizon, group in rows.groupby('horizon'):
            parts.append(
                pd.DataFrame(
                    {
                        'forecast': self.point[horizon].predict(group),
                        'lower': self.lower[horizon].predict(group),
                        'upper': self.upper[horizon].predict(group),
                    },
                    index=group.index,
                )
            )
        result = pd.concat(parts).loc[rows.index].clip(lower=0)
        result['lower'] = np.minimum(result['lower'], result['forecast'])
        result['upper'] = np.maximum(result['upper'], result['forecast'])
        return result


def fit_horizon(rows: pd.DataFrame) -> tuple[Pipeline, Pipeline, Pipeline]:
    x, y = split_x_y(rows)
    point = build_boosting().fit(x, y)
    lower = build_boosting('quantile', LOWER_QUANTILE).fit(x, y)
    upper = build_boosting('quantile', UPPER_QUANTILE).fit(x, y)
    return point, lower, upper


def fit_forecast_model(
    training_sets: dict[int, pd.DataFrame],
) -> ForecastModel:
    fitted = {
        horizon: fit_horizon(rows) for horizon, rows in training_sets.items()
    }
    last_answer = max(
        rows['target_date'].max() for rows in training_sets.values()
    )
    return ForecastModel(
        point={horizon: models[0] for horizon, models in fitted.items()},
        lower={horizon: models[1] for horizon, models in fitted.items()},
        upper={horizon: models[2] for horizon, models in fitted.items()},
        metadata={
            'trained_until': last_answer,
            'stores': sorted(
                {
                    int(store)
                    for rows in training_sets.values()
                    for store in rows['store_id'].unique()
                }
            ),
            'features': FEATURE_COLUMNS,
            'interval': (LOWER_QUANTILE, UPPER_QUANTILE),
            'versions': {
                'python': platform.python_version(),
                'scikit-learn': sklearn.__version__,
                'pandas': pd.__version__,
            },
        },
    )


def save_model(model: ForecastModel, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = MODEL_PATH) -> ForecastModel:
    if not path.exists():
        raise FileNotFoundError(f'Нет модели {path}. Обучить - make train')
    model = joblib.load(path)
    if not isinstance(model, ForecastModel):
        raise TypeError(f'В файле {path} не модель прогноза')
    return model


def forecast_week(
    model: ForecastModel,
    calendar: pd.DataFrame,
    plan: pd.DataFrame,
    store_id: int,
    start: pd.Timestamp,
) -> pd.DataFrame:
    stores = model.metadata.get('stores', [])
    if store_id not in stores:
        raise ValueError(
            f'Модель обучена только на магазинах {stores}, магазина {store_id} в ней нет'
        )
    origin = start - pd.Timedelta(days=1)
    frame = build_forecast_frame(calendar, plan, store_id, origin)
    rows = forecast_rows(frame, origin)
    forecast = model.predict(rows)

    days = frame.set_index('date')
    status = rows['target_date'].map(days['status']).to_numpy()
    closed = status == STATUS_CLOSE
    forecast.loc[closed, ['forecast', 'lower', 'upper']] = 0.0

    history = calendar[calendar['store_id'] == store_id].set_index('date')
    actual = rows['target_date'].map(history['target'])
    return pd.DataFrame(
        {
            'date': rows['target_date'],
            'horizon': rows['horizon'],
            'closed': closed,
            'forecast': forecast['forecast'].round(),
            'lower': forecast['lower'].round(),
            'upper': forecast['upper'].round(),
            'actual': actual,
            'error': forecast['forecast'].round() - actual,
        }
    ).reset_index(drop=True)
