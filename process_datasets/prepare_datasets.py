import numpy as np
import pandas as pd

def prepare_song_dataset(df_song):
    df_song_prepared = df_song
    df_song_prepared['genre_ids'] = df_song_prepared['genre_ids'].apply(lambda x: set(str(x).split('|')))
    df_song_prepared['artist_name'] = df_song_prepared['artist_name'].str.lower()
    df_song_prepared['composer'] = df_song_prepared['composer'].str.lower()
    df_song_prepared['lyricist'] = df_song_prepared['lyricist'].str.lower()
    df_song_prepared['language'] = df_song_prepared['language'].apply(lambda x: str(x))
    return df_song_prepared


def prepare_train_dataset(df_train):
    df_train_prepared = df_train

    return df_train_prepared


def prepare_user_dataset(df_user):
    df_user_prepared = df_user
    #range age
    df_user_prepared['bd'] = df_user_prepared['bd'].apply(lambda x: np.absolute(x))
    df_user_prepared['age_range'] = pd.cut(df_user_prepared['bd'], bins=[0,5,10,18,30,45,60,80,100])
    df_user_prepared['age_range'] =df_user_prepared['age_range'].astype('str')
    #transform datetime
    df_user_prepared['registration_init_time'] = pd.to_datetime(df_user_prepared['registration_init_time'], format="%Y%m%d")
    df_user_prepared['expiration_date'] = pd.to_datetime(df_user_prepared['expiration_date'], format="%Y%m%d")

    days = df_user_prepared.expiration_date - df_user_prepared.registration_init_time
    days = [d.days for d in days]
    df_user_prepared['days'] = days
    return df_user_prepared
