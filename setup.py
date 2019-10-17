"""A setuptools based setup module for DiviK algorithm."""

from setuptools import setup, find_packages
from spdivik import __version__

setup(
    name='divik',
    version=__version__,
    description='Divisive iK-means algorithm implementation',
    url='https://github.com/gmrukwa/divik',
    author='Grzegorz Mrukwa',
    author_email='g.mrukwa@gmail.com',
    classifiers=[
        # based on https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    entry_points={
        'console_scripts': [
            'divik=spdivik.__main__:main',
            'inspect=spdivik.inspect.__main__:main',
            'kmeans=spdivik.kmeans.__main__:main',
            'linkage=spdivik._linkage:main',
            'spectral=spdivik.spectral:main',
            'visualize=spdivik.visualize:main'
        ],
    },
    packages=find_packages(exclude=['test']),
    # @gmrukwa: https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        'dash==0.34.0',
        'dash-html-components==0.13.4',
        'dash-core-components==0.42.0',
        'dash-table==3.1.11',
        'functional-helpers',
        'h5py>=2.8.0',
        'numpy>=0.12.1',
        'pandas>=0.20.3',
        'scipy>=0.19.1',
        'scikit-image>=0.14.1',
        'scikit-learn>=0.19.1',
        'tqdm>=4.11.2',
        'typing>=3.6.2'
    ],
    extras_require={
        'all': [
            'matlabruntimeforpython===R2016b',
            'pyamg',
            'quilt>=2.9.12',
        ],
        'quilt_packages': ['quilt>=2.9.12'],
        'divik': ['matlabruntimeforpython===R2016b'],
        'spectral': ['pyamg']
    },
    python_requires='>=3.4,<3.6',
    package_data={
    }
)
