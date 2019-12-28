FROM python:3.7-slim

RUN apt-get update &&\
    apt-get install -y gcc &&\
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED TRUE

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

COPY . /app

RUN python dev_setup.py install

RUN python -m unittest discover
