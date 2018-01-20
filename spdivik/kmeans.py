"""Numpy-based implementation of k-means algorithm"""
from abc import ABCMeta, abstractmethod

import numpy as np

import spdivik.distance as dist


Labels = np.ndarray
Data = np.ndarray
Centroids = np.ndarray
