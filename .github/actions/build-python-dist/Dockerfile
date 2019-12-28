FROM python:3.7-slim

RUN apt-get update &&\
    apt-get install -y gcc &&\
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade \
    setuptools \
    wheel \
    twine

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
