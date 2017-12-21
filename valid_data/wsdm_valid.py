import pandas as pd
from process_features.CategoricalFeaturesExtraction import SimpleMultiLabelBinarizedFeatureExtraction, SimpleCategoricalFeatureExtraction
from process_datasets import prepare_datasets
from process_features.NumericalFeaturesExtraction import SimpleNumericalFeatureExtraction
from process_features.ProcessFeatures import ProcessFeatures
import lightgbm as lgbm


df_wsdm = pd.read_csv('/home/ignacio/Datasets/Music datasets/KKBOX/train_mini.csv')
df_song = pd.read_csv('/home/ignacio/Datasets/Music datasets/KKBOX/songs_cleaned.csv')
df_users = pd.read_csv('/home/ignacio/Datasets/Music datasets/KKBOX/members_cleaned.csv')

df_song = prepare_datasets.prepare_song_dataset(df_song)
df_train = prepare_datasets.prepare_train_dataset(df_wsdm)
df_users = prepare_datasets.prepare_user_dataset(df_users)

print(df_train.shape)
print(df_train.columns)

df_train = pd.merge(df_train, df_song, how='left', left_on='song_id', right_on='song_id')

df_train = pd.merge(df_train, df_users, how='left', left_on='msno', right_on='msno')

df_y_train = df_train['target']
df_train = df_train.drop('target',1)

print('Train dataset:')
print(df_train.shape)
print(df_train.columns)


#Get genres classes
genres_ids=list(df_song['genre_ids'])
classes = [x for genre in genres_ids for x in genre]
classes = set(classes)

#Concatenate train and test to get all labels for categorical features

numerical_feature_extraction = SimpleNumericalFeatureExtraction(col_to_extract=['song_length', 'freq_total', 'days'])
categorical_feature_extraction = SimpleCategoricalFeatureExtraction(col_to_extract=['source_system_tab', 'source_type',
                                                                                    'language', 'city','age_range','gender',
                                                                                    'registered_via', 'popularity'])

multilabel_feature_extraction = SimpleMultiLabelBinarizedFeatureExtraction(col_to_extract='genre_ids', classes=list(classes))

process_features = ProcessFeatures(list_features_extractions=[('numerical_features', numerical_feature_extraction),
                                                              ('categorical_features', categorical_feature_extraction),
                                                              ('multilabel_features', multilabel_feature_extraction)])



process_features.fit(df_train)
X_features_train = process_features.transform(df_train)


print('Train dataset:')
print(X_features_train.shape)


#cross fold validation

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'learning_rate': 0.2,
    'verbose': 0,
    'num_leaves': 2**8,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 128,
    'max_depth': 10,
    'num_rounds': 200,
    'metric': 'auc',
}

d_train = lgbm.Dataset(X_features_train, label= df_y_train)

eval_hist = lgbm.cv(params, train_set= d_train, nfold= 5)

df_eval_hist = pd.DataFrame(eval_hist)

print(df_eval_hist.head(10))
df_eval_hist.to_csv('evaluation_history.csv', index=False)
