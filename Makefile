RAW_DATA := data/raw/train.csv
CALENDAR := data/processed/calendar.csv

.PHONY: install download calendar data clean

install:
	uv sync

$(RAW_DATA):
	uv run python data/raw/download.py

download:
	uv run python data/raw/download.py

$(CALENDAR): $(RAW_DATA) src/data.py
	uv run python -m src.data

calendar: $(CALENDAR)

data: install calendar

clean:
	rm -f $(CALENDAR)
