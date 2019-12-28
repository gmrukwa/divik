#!/usr/bin/env bash

pip install --no-cache-dir -r requirements.txt

python setup.py sdist bdist bdist_egg bdist_wheel

ls dist
