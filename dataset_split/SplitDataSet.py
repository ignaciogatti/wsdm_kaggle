import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class SplitDataSet:

    def __init__(self, eval_size=0.10):
        self.eval_size = eval_size
        self.kf = StratifiedKFold( round(1./self.eval_size))

    def split(self,X,y):
        train_indices, test_indices = next(iter(self.kf.split(X,y)))
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        return X_train, y_train, X_test, y_test


