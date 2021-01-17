FROM python:3.7-slim AS base
ENV PYTHONUNBUFFERED TRUE
RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
WORKDIR /app
RUN apt-get update &&\
    apt-get install -y \
        libgomp1 \
        gcc \
        curl \
        git \
        ssh &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

COPY . /app
RUN poetry install
