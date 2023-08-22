FROM python:3.9-slim AS base
ENV PYTHONUNBUFFERED TRUE
RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
WORKDIR /app
RUN apt-get update &&\
    apt-get install -y libgomp1 &&\
    rm -rf /var/lib/apt/lists/*


FROM base as builder
SHELL ["/bin/bash", "-c"]
RUN apt-get update &&\
    apt-get install -y gcc curl &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${POETRY_HOME}/bin:${PATH}"
RUN poetry config virtualenvs.create false


FROM builder
ENV ENABLE_SLOW_TESTS True
COPY requirements-dev.txt requirements-dev.txt
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install
COPY . /app
RUN poetry run pip freeze | grep numpy
RUN poetry install
RUN poetry build
RUN poetry run pytest
