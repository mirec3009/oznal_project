import pandas as pd
import numpy as np

def split_data(df):
    from sklearn.model_selection import train_test_split
    
    df_sorted = df.sort_values(by='date')
    X = df_sorted.loc[:, df_sorted.columns != 'price']
    y = df_sorted['price']

    #split 70:20:10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.33, shuffle=False)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def replace_outliers(value, l_whisker, r_whisker, quantile_05, quantile_95):
    new_value = value

    if value < l_whisker:
        new_value = quantile_05
    elif value > r_whisker:
        new_value = quantile_95

    return new_value


def boxcox_normalize(col_train, col_valid, col_test):
    from src.analysis import get_whiskers
    import scipy.stats as stats
    
    col_train_norm, att = stats.boxcox(col_train)
    col_valid_norm = stats.boxcox(col_valid, att)
    col_test_norm = stats.boxcox(col_test, att)
    
    col_train_norm = pd.Series(col_train_norm)
    col_valid_norm = pd.Series(col_valid_norm)
    col_test_norm = pd.Series(col_test_norm)
    
    l_whisker, r_whisker = get_whiskers(col_train_norm)
    train_05 = col_train_norm.quantile(0.05)
    train_95 = col_train_norm.quantile(0.95)
    
    col_train = col_train_norm.apply(replace_outliers, args=(l_whisker, r_whisker, train_05, train_95))
    col_valid = col_valid_norm.apply(replace_outliers, args=(l_whisker, r_whisker, train_05, train_95))
    col_test = col_test_norm.apply(replace_outliers, args=(l_whisker, r_whisker, train_05, train_95))

    return np.array(col_train), np.array(col_valid), np.array(col_test)


def replace_bedrooms_number(old_value, new_value, df):
    df.loc[df['bedrooms']==old_value,'bedrooms'] = new_value
    
    return df


def remove_rows(df):
    df = df.drop(df[(df['bedrooms']==0)&(df['bathrooms']==0.0)].index)

    return df


def run_pipeline(df):
    df = replace_bedrooms_number(33, 3, df)
    df = remove_rows(df)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    
    #normalize price
    y_train, y_valid, y_test = boxcox_normalize(y_train, y_valid, y_test)
    
    #normalize sqft_living
    normalized = boxcox_normalize(X_train['sqft_living'], X_valid['sqft_living'], X_test['sqft_living'])
    X_train['sqft_living'] = normalized[0]
    X_valid['sqft_living'] = normalized[1]
    X_test['sqft_living'] = normalized[2]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test



