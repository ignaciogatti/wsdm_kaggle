import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

#Extract features from a model (e.x Kmeans)
class ModelTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))

