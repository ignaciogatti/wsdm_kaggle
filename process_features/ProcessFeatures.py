from sklearn.pipeline import FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator


class ProcessFeatures(TransformerMixin, BaseEstimator):

    def __init__(self, list_features_extractions):
        self.list_features_extractions = list_features_extractions


    def fit(self, X, y=None):
        self.feat_union_ = FeatureUnion(transformer_list=self.list_features_extractions)
        return self.feat_union_.fit(X,y)

    def transform(self, X):
        return self.feat_union_.transform(X)

    def get_params(self, deep=True):
        return {'list_features_extractions':self.list_features_extractions}

    ##TODO: father class
    def transform_valid(self, X_test):
        for transformer in self.list_features_extractions:
            transformer.transform_test(X_test)

