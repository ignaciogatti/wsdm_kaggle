import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from scipy import sparse

class ToDenseMatrixTransformer(TransformerMixin):

    def transform(self,X):
        if sparse.issparse(X):
            X = X.toarray()
        return X

    def fit(self, X, y=None):
        return self