import pandas as pd
from process_features.CategoricalFeaturesExtraction import SimpleMultiLabelBinarizedFeatureExtraction, SimpleCategoricalFeatureExtraction
from process_datasets import prepare_datasets
from process_features.NumericalFeaturesExtraction import SimpleNumericalFeatureExtraction
from process_features.ProcessFeatures import ProcessFeatures
import lightgbm as lgbm


df_wsdm = pd.read_csv('/home/ignacio/Datasets/Music datasets/KKBOX/train_sampled.csv')
df_song = pd.read_csv('/home/ignacio/Datasets/Music datasets/KKBOX/songs_cleaned.csv')
df_users = pd.read_csv('/home/ignacio/Datasets/Music datasets/KKBOX/members_cleaned.csv')

df_song = prepare_datasets.prepare_song_dataset(df_song)
df_train = prepare_datasets.prepare_train_dataset(df_wsdm)
df_users = prepare_datasets.prepare_user_dataset(df_users)

#merged train set
df_train = pd.merge(df_train, df_song, how='left', left_on='song_id', right_on='song_id')
df_train = pd.merge(df_train, df_users, how='left', left_on='msno', right_on='msno')

df_y_train = df_train['target']
df_train = df_train.drop('target', 1)

# Get genres classes
genres_ids = list(df_song['genre_ids'])
classes = [x for genre in genres_ids for x in genre]
classes = set(classes)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'learning_rate': 0.1,
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

i= 0
while i<5:
    test_path = '/home/ignacio/Datasets/Music datasets/KKBOX/test_cleaned_'+str(i)+'.csv'
    print('Round: ' + str(i))

    df_test = pd.read_csv(test_path)
    df_test = prepare_datasets.prepare_train_dataset(df_test)

    #merged test set
    df_test = pd.merge(df_test, df_song, how='left', left_on='song_id', right_on='song_id')
    df_test = pd.merge(df_test, df_users, how='left', left_on='msno', right_on='msno')


    print('Train dataset:')
    print(df_train.shape)
    print(df_train.columns)

    train_tail = df_train.shape[0]

    print('Test dataset:')
    print(df_test.shape)


    # Concatenate train and test to get all labels for categorical features
    df_test_to_concat = df_test.drop('id', 1)

    df_train_appended = df_train.append(df_test_to_concat)

    print('Appended data:')
    print(df_train.shape)

    numerical_feature_extraction = SimpleNumericalFeatureExtraction(
        col_to_extract=['song_length', 'freq_total', 'days'])

    categorical_feature_extraction = SimpleCategoricalFeatureExtraction(
        col_to_extract=['source_system_tab', 'source_type',
                        'language', 'city', 'age_range', 'gender',
                        'registered_via','popularity'])
    '''
        categorical_feature_extraction = SimpleCategoricalFeatureExtraction(
            col_to_extract=['source_system_tab', 'source_screen_name',
                            'source_type', 'artist_name', 'composer',
                            'lyricist', 'language', 'city', 'bd',
                            'gender', 'registered_via'])
    '''
    multilabel_feature_extraction = SimpleMultiLabelBinarizedFeatureExtraction(col_to_extract='genre_ids',
                                                                               classes=list(classes))

    process_features = ProcessFeatures(list_features_extractions=[('numerical_features', numerical_feature_extraction),
                                                                  ('categorical_features',
                                                                   categorical_feature_extraction),
                                                                  ('multilabel_features',
                                                                   multilabel_feature_extraction)])

    process_features.fit(df_train_appended)
    X_features = process_features.transform(df_train_appended)

    print(X_features.shape)

    X_features_train = X_features[0:train_tail][:]

    X_features_test = X_features[train_tail:][:]

    print('Train dataset:')
    print(X_features_train.shape)
    print('Test dataset:')
    print(X_features_test.shape)

    d_train = lgbm.Dataset(X_features_train, label=df_y_train)

    watchlist = [d_train]

    print('Training LGBM model...')
    model = lgbm.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, early_stopping_rounds=10,
                       verbose_eval=10)

    model.save_model('/home/ignacio/Datasets/Music datasets/KKBOX/lgbm_model')
    model.set_train_data_name('/home/ignacio/Datasets/Music datasets/KKBOX/training_dataset')

    print('Making predictions and saving them...')
    p_test = model.predict(X_features_test)

    subm = pd.DataFrame()
    subm['id'] = df_test['id']
    subm['target'] = p_test
    submission_path = '/home/ignacio/Datasets/Music datasets/KKBOX/submission_'+ str(i) +'.csv.gz'
    subm.to_csv(submission_path, compression='gzip', index=False,
                float_format='%.5f')

    i += 1
    print('Done!')

