import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from process_features.ColumnExtractor import ColumnExtractor
from abc import ABC, abstractclassmethod

class TextFeaturesExtraction(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, cols_to_extract ):
        self.col_extract = cols_to_extract
        self.tfidvec = TfidfVectorizer()


    @abstractclassmethod
    def process_data(self, X):
        pass

    def fit(self, X, y=None):
        self.col_extracted_ = ColumnExtractor(self.col_extract)
        return self

    def transform(self, X ):
        self.col_extracted_.fit(X)
        X_text = self.col_extracted_.transform(X)
        text = self.process_data(X_text)
        self.tfidvec.fit(text)
        X_text_tfidf = self.tfidvec.transform( text )
        save_npz('text_features.npz', X_text_tfidf)
        return X_text_tfidf


    def get_params(self, deep=True):
        return {'cols_to_extract':self.col_extract}

    def get_transformers(self):
        return {'tfidf':self.tfidvec}


class TextFeatureExtractionTitanic(TextFeaturesExtraction):

    def process_data(self, X):
        text = [ str(X[i, 0] + X[i, 1]) for i in range( X.shape[0]) ]
        return text