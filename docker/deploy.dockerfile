FROM python:3.7-slim AS base
ENV PYTHONUNBUFFERED TRUE
RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
WORKDIR /app
RUN apt-get update &&\
    apt-get install -y libgomp1 &&\
    rm -rf /var/lib/apt/lists/*


FROM base as builder
RUN mkdir -p /install/lib/python3.7/site-packages
ENV PYTHONPATH .:/install/lib/python3.7/site-packages
RUN apt-get update &&\
    apt-get install -y gcc &&\
    rm -rf /var/lib/apt/lists/*


FROM builder AS deps_builder
COPY requirements.txt requirements.txt
RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    --no-warn-script-location \
    -r requirements.txt


FROM builder as divik_builder
COPY --from=deps_builder /install /usr/local
COPY . /app
RUN python setup.py install --prefix=/install


FROM base
EXPOSE 8050
VOLUME /data
WORKDIR /data
COPY --from=deps_builder /install /usr/local
COPY --from=divik_builder /install /usr/local
