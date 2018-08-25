"""A setuptools based setup module for DiviK algorithm."""

from setuptools import setup, find_packages
from spdivik import __version__

setup(
    name='spectre-divik',
    version=__version__,
    description='Divisive iK-means algorithm implementation',
    url='https://github.com/spectre-team/spectre-divik',
    author='Grzegorz Mrukwa',
    author_email='Grzegorz.Mrukwa@polsl.pl',
    classifiers=[
        # based on https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    entry_points={
        'console_scripts': [
            'divik=spdivik.__main__:main',
            'kmeans=spdivik.kmeans.__main__:main',
            'linkage=spdivik.linkage:main'
        ],
    },
    packages=find_packages(exclude=['test']),
    # @gmrukwa: https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        'functional-helpers',
        'matlabruntimeforpython===R2016b',
        'numpy>=0.12.1',
        'pandas<0.21',
        'scipy>=0.19.1',
        'tqdm>=4.11.2',
        'typing>=3.6.2'
    ],
    python_requires='==3.4.*',
    package_data={
    }
)
