FROM python:3.7-slim

RUN apt-get update &&\
    apt-get install -y gcc &&\
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED TRUE

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
  rm /requirements.txt

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

WORKDIR /app

COPY gamred_native /app/gamred_native

COPY dev_setup.py /app/dev_setup.py

RUN python dev_setup.py install &&\
    rm -rf /app/gamred_native &&\
    rm /app/dev_setup.py
