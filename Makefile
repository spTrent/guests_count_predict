RAW_DATA := data/raw/train.csv
RAW_PLAN := data/raw/test.csv
CALENDAR := data/processed/calendar.csv
DATE ?= 2015-08-01
STORE ?= 1
IMAGE ?= guests-forecast
MODEL := models/model.joblib

.PHONY: install download calendar data train predict test lint requirements docker-build docker-run clean

install:
	uv sync

$(RAW_DATA):
	uv run python data/raw/download.py

$(RAW_PLAN): $(RAW_DATA)

download:
	uv run python data/raw/download.py

$(CALENDAR): $(RAW_DATA) $(RAW_PLAN) src/data.py
	uv run python -m src.data

calendar: $(CALENDAR)

data: install calendar

train: $(CALENDAR)
	uv run python -m src.train

predict:
	uv run python predict.py --date $(DATE) --store $(STORE)

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src predict.py

requirements:
	uv export --format requirements-txt --no-hashes --no-header --no-annotate --all-groups --no-emit-project -o requirements.txt

$(MODEL): $(CALENDAR)
	uv run python -m src.train

docker-build: $(MODEL)
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm $(IMAGE) --date $(DATE) --store $(STORE)

clean:
	rm -f $(CALENDAR) data/processed/plan.csv
