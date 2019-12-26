FROM spectreteam/python_msi:v5.1.0.2019a.py37

RUN apt-get update &&\
    apt-get install -y gcc &&\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
  rm /requirements.txt

RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

RUN apt-get update && apt-get install -y --no-install-recommends \
		make \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sphinx numpydoc sphinx-rtd-theme
