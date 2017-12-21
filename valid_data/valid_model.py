import pandas as pd
import numpy as np

class Valid_Model:

    def __init__(self, transformers):
        self.transformers = transformers


    def transform_valid_data(self, X_valid):
        X_processed = []
        for t in self.transformers:
            X_processed.append(t.transform_test(X_valid))
        X_transformed = np.hstack(X_processed)
        return X_transformed


