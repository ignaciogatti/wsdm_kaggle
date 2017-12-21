import pandas as pd
import numpy as np
from process_features.CategoricalFeaturesExtraction import SimpleCategoricalFeatureExtraction
from process_features.TextFeaturesExtraction import TextFeatureExtractionTitanic
from process_features.NumericalFeaturesExtraction import SimpleNumericalFeatureExtraction
from process_features.ProcessFeatures import ProcessFeatures
from process_features.DenseMatrixTransformer import ToDenseMatrixTransformer
from feature_selection.FeatureSelection import PCAFeatureSelection
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from ml_model_selector.ml_pipeline import MLModelPipeline
from dataset_split.SplitDataSet import SplitDataSet

columns_to_keep =['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 'Embarked']

df_train = pd.read_csv('/home/ignacio/PycharmProjects/MLFramework/train.csv')

y = df_train['Survived']
X = df_train[columns_to_keep]
print('Full dataset')
print(X.shape)

split_data = SplitDataSet(0.10)
X_train, y_train, X_test, y_test = split_data.split(X,y)
print('Train dataset')
print(X_train.shape)

cat_feat_extraction = SimpleCategoricalFeatureExtraction(col_to_extract=['Pclass', 'Sex', 'Embarked'])

text_feat_extraction = TextFeatureExtractionTitanic(cols_to_extract=['Name', 'Sex'])

numerical_feat_extraction = SimpleNumericalFeatureExtraction(col_to_extract=['Age', 'SibSp', 'Parch', 'Fare'])

process_features = ProcessFeatures(list_features_extractions=
                                   [('categorical_features', cat_feat_extraction),('text_features', text_feat_extraction),
                                    ('numerical_features', numerical_feat_extraction)])

pca = PCAFeatureSelection(12)

clf = RandomForestClassifier()

model_pipeline = MLModelPipeline(clf=clf, process_features=process_features, feature_selection=pca)
model_pipeline.set_params_feature_selection(**{'n_components':[8,10,12]})
model_pipeline.set_params_estimator(**{'n_estimators':[120, 300], 'max_depth':[5, 8, 15]})

estimator = model_pipeline.get_best_estimator('accuracy',X_train,y_train)

print(estimator.best_params_)

df_results = pd.DataFrame(estimator.cv_results_)
df_results.to_csv('results_estimator_analisis.csv')

print(df_results.shape)
print(list(df_results))
