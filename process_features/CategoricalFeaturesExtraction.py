import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from scipy.sparse import save_npz
from process_features.ColumnExtractor import ColumnExtractor
from abc import ABC, abstractclassmethod


class CategoricalFeatureExtraction(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, col_to_extract=None, labels = None):
        self.col_extract = col_to_extract
        self.lb_enc = []
        self.labels = labels
        self.ohe = OneHotEncoder()

    @abstractclassmethod
    def process_data(self, X):
        pass

    def fit(self, X, y=None):
        self.col_extracted_ = ColumnExtractor(self.col_extract)
        return self

    def transform(self, X):
        self.col_extracted_.fit(X)
        X_categorical = self.col_extracted_.transform(X)
        X_categorical_processed = self.process_data(X_categorical)
        n_features = X_categorical_processed.shape[1]
        for i in range(0,n_features):
            lb = LabelEncoder()
            if self.labels is not None:
                lb.fit(self.labels[i])
            else:
                lb.fit(X_categorical[:,i])
            X_categorical_processed[:, i] = lb.transform(X_categorical_processed[:,i])
            self.lb_enc.append(lb)

        self.ohe.fit(X_categorical_processed)
        X_categorical_labeled = self.ohe.transform(X_categorical_processed)
        save_npz('categorical_features.npz', X_categorical_labeled )
        return X_categorical_labeled

    def get_params(self, deep=True):
        return {'col_to_extract':self.col_extract}

    def set_params(self, **kwargs):
        for key,value in kwargs:
            setattr(self, key, value)

    def get_transformers(self):
        return {'label_encoders':self.lb_enc, 'one_hot_encoder':self.ohe}

    def transform_test(self, X_test):
        self.col_extracted_.fit(X_test)
        X_test_categorical = self.col_extracted_.transform(X_test)
        n_features = X_test_categorical.shape[1]
        for i in range(0, n_features):
            X_test_categorical[:,i] = self.lb_enc[i].transform(X_test_categorical[:,i])
        X_test_labeled = self.ohe.transform(X_test_categorical)
        return X_test_labeled



class MultiLabelBinarizedFeatureExtraction(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, col_to_extract=None, classes=None):
        self.col_to_extract = col_to_extract
        self.mlb = MultiLabelBinarizer(classes=classes)
        self.ohe = OneHotEncoder()

    def fit(self, X, y=None):
        self.col_extracted_ = ColumnExtractor(self.col_to_extract)
        return self

    def transform(self, X):
        self.col_extracted_.fit(X)
        X_labeled = self.col_extracted_.transform(X)
        self.mlb.fit(X_labeled)
        X_labeled_processed = self.mlb.transform(X_labeled)

        self.ohe.fit(X_labeled_processed)
        X_labeled_processed = self.ohe.transform(X_labeled_processed)
        return X_labeled_processed

    @abstractclassmethod
    def process_data(self, X):
        pass

    def transform_test(self, X_test):
        self.col_extracted_.fit(X_test)
        X_test_labeled = self.col_extracted_.transform(X_test)
        X_test_multilabeled = self.mlb.transform(X_test_labeled)
        X_test_multilabeled_processed = self.ohe.transform(X_test_multilabeled)
        return X_test_multilabeled_processed


class SimpleMultiLabelBinarizedFeatureExtraction(MultiLabelBinarizedFeatureExtraction):

    def process_data(self, X):
        return X


class SimpleCategoricalFeatureExtraction(CategoricalFeatureExtraction):

    def process_data(self, X):
        return X


