FROM spectreteam/python_msi:v5.0.0

RUN pip install --no-cache-dir --upgrade \
    setuptools \
    wheel \
    twine

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
