#!/usr/bin/env bash

/opt/python/cp35-cp35m/bin/pip install -r requirements-base.txt
/opt/python/cp35-cp35m/bin/python setup.py sdist
/opt/python/cp35-cp35m/bin/pip wheel . -w dist/tmp
auditwheel repair dist/tmp/divik*whl -w dist
rm -rf dist/tmp

/opt/python/cp36-cp36m/bin/pip install -r requirements-base.txt
/opt/python/cp36-cp36m/bin/pip wheel . -w dist/tmp
auditwheel repair dist/tmp/divik*whl -w dist
rm -rf dist/tmp

/opt/python/cp37-cp37m/bin/pip install -r requirements-base.txt
/opt/python/cp37-cp37m/bin/pip wheel . -w dist/tmp
auditwheel repair dist/tmp/divik*whl -w dist
rm -rf dist/tmp

/opt/python/cp38-cp38/bin/pip install -r requirements-base.txt
/opt/python/cp38-cp38/bin/pip wheel . -w dist/tmp
auditwheel repair dist/tmp/divik*whl -w dist
rm -rf dist/tmp

ls dist
