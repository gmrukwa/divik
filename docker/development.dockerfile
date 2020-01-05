FROM python:3.7-slim AS base
ENV PYTHONUNBUFFERED TRUE
RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
WORKDIR /app


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


FROM builder as gamred_builder
COPY --from=deps_builder /install /usr/local
COPY dev_setup.py dev_setup.py
COPY gamred_native gamred_native
RUN python dev_setup.py install --prefix=/install


FROM base
COPY --from=deps_builder /install /usr/local
COPY --from=gamred_builder /install /usr/local
COPY . /app
