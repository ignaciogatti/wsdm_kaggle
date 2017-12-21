import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import six



class MLModelPipeline:

    def __init__(self, process_features, feature_selection, clf):
        self.process_features = process_features
        self.feature_selection = feature_selection
        self.clf = clf
        self.params = {}


    def define_pipeline(self):
        pipe = Pipeline([('process_features', self.process_features),('feature_selection', self.feature_selection),
                         ('estimator', self.clf)])
        return pipe


    def set_params_feature_selection(self, **params_feature_selection):
        if not params_feature_selection:
            return self
        for key, value in six.iteritems(params_feature_selection):
            key_new = 'feature_selection__' + key
            self.params.update({key_new:value})
        return self


    def set_params_estimator(self, **params_estimator):
        if not params_estimator:
            return self
        for key, value in six.iteritems(params_estimator):
            key_new = 'estimator__' + key
            self.params.update({key_new:value})
        return self


    def get_best_estimator(self, score, X_train, y_train):
        pipe = self.define_pipeline()
        grid = GridSearchCV(estimator=pipe, param_grid=self.params, scoring=score)
        print(self.params)
        grid.fit(X_train, y_train)
        return grid



