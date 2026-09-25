import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data import PLAN_PATH, PROCESSED_PATH, load_calendar, load_plan
from src.model import MODEL_PATH, ForecastModel, forecast_week, load_model

DAY_NAMES = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
OUTPUT_COLUMNS = {
    'date': 'Дата',
    'day': 'День',
    'forecast': 'Прогноз',
    'lower': 'От',
    'upper': 'До',
    'note': 'Комментарий',
    'actual': 'Факт',
    'error': 'Ошибка',
}


def parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(value, format='%Y-%m-%d')
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'неверная дата {value!r}, ожидается ГГГГ-ММ-ДД'
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Прогноз числа покупателей магазина на 7 дней вперёд'
    )
    parser.add_argument(
        '--date',
        type=parse_date,
        required=True,
        help='первый день прогноза, ГГГГ-ММ-ДД',
    )
    parser.add_argument(
        '--store',
        '--restaurant',
        dest='store',
        type=int,
        required=True,
        help='номер магазина',
    )
    parser.add_argument(
        '--output', type=Path, help='сохранить прогноз в CSV-файл'
    )
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--calendar', type=Path, default=PROCESSED_PATH)
    parser.add_argument('--plan', type=Path, default=PLAN_PATH)
    return parser


def format_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    return (
        forecast.assign(
            date=forecast['date'].dt.strftime('%Y-%m-%d'),
            day=forecast['date'].dt.dayofweek.map(dict(enumerate(DAY_NAMES))),
            note=forecast['closed'].map(
                {True: 'закрыт по графику', False: ''}
            ),
        )[list(OUTPUT_COLUMNS)]
        .rename(columns=OUTPUT_COLUMNS)
        .astype({'Прогноз': int, 'От': int, 'До': int})
        .astype({'Факт': 'Int64', 'Ошибка': 'Int64'})
        .astype({'Факт': 'string', 'Ошибка': 'string'})
        .fillna({'Факт': '', 'Ошибка': ''})
    )


def holdout_lines(model: ForecastModel, store_id: int) -> list[str]:
    metrics = model.metadata.get('holdout_metrics')
    if not metrics:
        return ['Метрики holdout в модели не сохранены']
    lines = [
        'Качество модели на отложенном периоде '
        f'{model.metadata["holdout_start"]} - {model.metadata["holdout_end"]}:'
    ]
    for scope, label in [
        (str(store_id), f'магазин {store_id}'),
        ('все', 'все магазины'),
    ]:
        boosting = metrics['boosting'][scope]
        baseline = metrics['baseline_two_weeks_ago'][scope]
        lines.append(
            f'  {label}: MAE {boosting["MAE"]} покупателя в день, '
            f'MAPE {boosting["MAPE, %"]}% '
            f'(бейзлайн «тот же день две недели назад»: '
            f'MAE {baseline["MAE"]}, MAPE {baseline["MAPE, %"]}%)'
        )
    lower, upper = model.metadata['interval']
    lines.append(
        f'  интервал между квантилями {lower:.0%} и {upper:.0%} '
        f'накрыл {model.metadata["holdout_coverage"]}% фактов'
    )
    return lines


def actual_lines(model: ForecastModel, forecast: pd.DataFrame) -> list[str]:
    known = forecast[~forecast['closed'] & forecast['actual'].notna()]
    if known.empty:
        return []
    mae = known['error'].abs().mean()
    mape = (known['error'].abs() / known['actual']).mean() * 100
    lines = [
        f'Факт по этим дням известен ({len(known)} открытых дней): '
        f'MAE {mae:.1f}, MAPE {mape:.1f}%'
    ]
    trained_until = model.metadata.get('trained_until')
    if trained_until is not None and known['date'].min() <= trained_until:
        lines.append(
            '  модель обучалась на данных по '
            f'{trained_until.date()}, включая эти дни, поэтому эта оценка '
            'завышена. Честная оценка - метрики holdout выше'
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = load_model(args.model)
        forecast = forecast_week(
            model,
            load_calendar(args.calendar),
            load_plan(args.plan),
            args.store,
            args.date,
        )
    except (FileNotFoundError, ValueError, TypeError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 1

    end = forecast['date'].max().date()
    print(
        f'Магазин {args.store}: прогноз покупателей на {args.date.date()} - {end}'
    )
    print(format_forecast(forecast).to_string(index=False, na_rep=''))
    print()
    print('\n'.join(holdout_lines(model, args.store)))
    extra = actual_lines(model, forecast)
    if extra:
        print()
        print('\n'.join(extra))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(args.output, index=False, date_format='%Y-%m-%d')
        print(f'\nПрогноз сохранён: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
