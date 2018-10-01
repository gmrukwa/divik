FROM spectreteam/python_msi:v4.0.9

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip

RUN apt-get -qq update &&\
  apt-get -qq install git &&\
  pip install -r requirements.txt &&\
  apt-get -qq remove git &&\
  apt-get autoremove -y &&\
  rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

RUN python -m unittest discover

RUN python setup.py install

VOLUME /data

WORKDIR /data

RUN rm -rf /app
