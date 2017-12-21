import numpy as np
import pandas as pd
from abc import ABC, abstractclassmethod
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.decomposition import PCA


class FeatureSelection(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, n_components):
        self.n_components = n_components
        self.feature_selector = None

    @abstractclassmethod
    def process_data(self,X):
        pass

    def transform(self,X):
        X_processed = self.process_data(X)
        self.feature_selector.fit(X_processed)
        return self.feature_selector.transform(X_processed)

    def fit(self,X, y=None):
        return self.feature_selector.fit(X, y)

    def get_params(self, deep=True):
       return {'n_components':self.n_components}


    def get_transformer(self):
        return {'feature_selection':self.feature_selector}



class PCAFeatureSelection(FeatureSelection):

    def __init__(self, n_components):
        super(PCAFeatureSelection, self).__init__(n_components)
        self.feature_selector = PCA(n_components=n_components)

    def process_data(self,X):
        X_dense = X.toarray()
        X_dense = np.nan_to_num(X_dense)
        return X_dense

    def fit(self,X, y=None):
        return self

