FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY predict.py ./
COPY models/model.joblib ./models/model.joblib
COPY data/processed/calendar.csv data/processed/plan.csv ./data/processed/

ENTRYPOINT ["/app/.venv/bin/python", "predict.py"]
CMD ["--date", "2015-08-01", "--store", "1"]
