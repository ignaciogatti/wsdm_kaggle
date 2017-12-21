import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from process_features.DenseMatrixTransformer import ToDenseMatrixTransformer
from abc import ABC
from sklearn.preprocessing import Normalizer

class NormalizeFeatures(ABC, BaseEstimator, TransformerMixin):

    def __init__(self):
        self.dense_transformer = ToDenseMatrixTransformer()
        self.normalizer = None

    def fit(self, X, y=None):
        return self

    def transform(self,X):
        self.dense_transformer.fit(X)
        X_dense = self.transform(X)
        return X_dense


class Normalizer_Matrix(NormalizeFeatures):

    def __init__(self):
        self.normalizer = Normalizer()
