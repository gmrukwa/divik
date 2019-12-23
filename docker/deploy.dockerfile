FROM spectreteam/python_msi:v5.1.0.2019a.py37

ENV PYTHONUNBUFFERED TRUE

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

COPY . /app

RUN python setup.py install

EXPOSE 8050

VOLUME /data

WORKDIR /data

RUN rm -rf /app
