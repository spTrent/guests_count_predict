# Прогноз потока гостей ресторана

## Данные

**Источник:** соревнование Kaggle [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales). Это дневные продажи 1115 магазинов сети Rossmann. Число посетителей за день (`Customers`) используется как аналог числа гостей, выручка (`Sales`) — как выручка точки.

| Файл | Строк | Период | Что внутри |
|---|---|---|---|
| `train.csv` | 1 017 209 | 2013-01-01 … 2015-07-31 | `Store`, `Date`, `Sales`, `Customers`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` |
| `test.csv` | 41 088 | 2015-08-01 … 2015-09-17 | те же признаки, но **без** `Sales` и `Customers` |

В `test.csv` нет целевой переменной, поэтому качество модели на нём не измерить. Отложенная выборка для валидации берётся из последних недель `train.csv`.

Сами CSV в репозиторий не коммитятся (`train.csv` весит 38 МБ) и лежат в `.gitignore`. Вместо них в репозитории есть скрипт скачивания.

### Как скачиваются данные

Скрипт [`data/raw/download.py`](data/raw/download.py):

1. берёт токен Kaggle из `.env` через `python-dotenv`;
2. через `kagglehub` запрашивает у Kaggle только `train.csv` и `test.csv`, поэтому `store.csv` и `sample_submission.csv` не скачиваются;
3. распаковывает архивы прямо в `data/raw/` и удаляет zip-файлы;
4. удаляет из `data/raw/` всё лишнее, в том числе служебную папку `.complete`, которую создаёт `kagglehub`.

Скачивание всегда идёт заново (`force_download=True`), поэтому повторный запуск перезаписывает файлы и не падает.

### Как воспроизвести

1. Получите API-токен Kaggle: *kaggle.com → Settings → API → Create New Token*.
2. Примите правила соревнования на [странице Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/rules). Без этого Kaggle вернёт ошибку 403.
3. Создайте в корне репозитория файл `.env`:

   ```
   KAGGLE_API_TOKEN=<ваш токен>
   ```

4. Установите зависимости и запустите скачивание:

   ```bash
   uv sync
   uv run python data/raw/download.py
   ```

После запуска в `data/raw/` будут только `train.csv`, `test.csv`, скрипт и `.gitkeep`.
