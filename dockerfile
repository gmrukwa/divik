FROM spectreteam/python_msi:v5.0.0 as base


FROM base as builder

RUN mkdir /install

WORKDIR /install

COPY requirements.txt /requirements.txt

RUN pip install --install-option="--prefix=/install" -r /requirements.txt


FROM base

COPY --from=builder /install /usr/local

COPY . /app

WORKDIR /app

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

RUN python -m unittest discover

RUN python setup.py install

VOLUME /data

WORKDIR /data

RUN rm -rf /app
