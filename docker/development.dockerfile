FROM spectreteam/python_msi:v5.1.0.2019a.py37

ENV PYTHONUNBUFFERED TRUE

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
  rm /requirements.txt

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
