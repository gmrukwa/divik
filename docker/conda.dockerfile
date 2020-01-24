FROM continuumio/miniconda3

ENV PYTHONUNBUFFERED TRUE

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

RUN apt-get update &&\
    apt-get install -y gcc &&\
    rm -rf /var/lib/apt/lists/*

RUN conda install -y \
    matplotlib \
    numpy \
    pandas \
    python=3.7 \
    scikit-learn \
    scikit-image \
    scipy \
    tqdm

COPY requirements-base.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

COPY . /app

RUN pip install --no-cache-dir /app \
    && rm -rf /app

VOLUME /data

WORKDIR /data
