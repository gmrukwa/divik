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
RUN mkdir -p /install/lib/python3.9/site-packages
ENV PYTHONPATH .:/install/lib/python3.9/site-packages
RUN apt-get update &&\
    apt-get install -y gcc curl &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${POETRY_HOME}/bin:${PATH}"


FROM builder AS deps_builder
COPY . /app
RUN poetry install &&\
    poetry build


FROM builder AS deps_install
COPY --from=deps_builder /app/dist /app/dist
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    --prefix=/install \
    --no-cache-dir \
    --no-warn-script-location &&\
    pip install /app/dist/divik*.whl \
    --prefix=/install \
    --no-cache-dir \
    --no-warn-script-location


FROM base
EXPOSE 8050
VOLUME /data
WORKDIR /data
COPY --from=deps_install /install /usr/local
