FROM spectreteam/python_msi:v5.1.0.2019a.py37

RUN pip install --no-cache-dir --upgrade \
    setuptools \
    wheel \
    twine

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
