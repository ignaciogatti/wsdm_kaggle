import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from scipy import sparse
from scipy.sparse import save_npz
from process_features.ColumnExtractor import ColumnExtractor
from abc import ABC, abstractclassmethod


class NumericalFeatureExtraction(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, col_to_extract):
        self.col_extract = col_to_extract

    @abstractclassmethod
    def process_data(self, X):
        pass

    def fit(self, X, y=None):
        self.col_extracted_ = ColumnExtractor(self.col_extract)
        return self

    def transform(self, X):
        self.col_extracted_.fit(X)
        X_numerical = self.col_extracted_.transform(X)
        X_numerical_processed = self.process_data(X_numerical)
        X_numerical_sparsed = sparse.csr_matrix(X_numerical_processed)
        save_npz('numerical_features.npz', X_numerical_sparsed)
        return X_numerical_sparsed


    def get_params(self, deep=True):
        return {'col_to_extract': self.col_extract}

    def get_transformer(self):
        return {}

    def transform_test(self, X_test):
        self.col_extracted_.fit(X_test)
        X_test_numerical = self.col_extracted_.transform(X_test)
        return X_test_numerical


class SimpleNumericalFeatureExtraction(NumericalFeatureExtraction):

    def process_data(self, X):
        return X

