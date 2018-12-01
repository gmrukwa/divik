FROM spectreteam/python_msi:v5.0.0

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

RUN python -m unittest discover

RUN python setup.py install

VOLUME /data

WORKDIR /data

RUN rm -rf /app
