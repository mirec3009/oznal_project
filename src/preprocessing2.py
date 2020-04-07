import pandas as pd
import numpy as np
import category_encoders as ce
import scipy.stats as stats
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from src.analysis import get_whiskers, calc_recon_age
    

def split_data(df):
    df_sorted = df.sort_values(by='date')
    X = df_sorted.loc[:, df_sorted.columns != 'price']
    y = df_sorted['price']

    #split 70:20:10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.33, shuffle=False)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def encode_zipcode(df):
    encoder = ce.binary.BinaryEncoder(cols=['zipcode'])
    df_encoded = encoder.fit_transform(df)
    
    return df_encoded


def get_values_for_replacement(column, method='5-95perc'):
    if method == '5-95perc':
        value1 = column.quantile(0.05)
        value2 = column.quantile(0.95)
    elif method == 'mean':
        value1 = value2 = column.mean()
    elif method == 'med':
        value1 = value2 = column.median()
        
    return value1, value2


def replace_outliers(value, l_whisker, r_whisker, quantile_05, quantile_95):
    new_value = value

    if value < l_whisker:
        new_value = quantile_05
    elif value > r_whisker:
        new_value = quantile_95

    return new_value


def boxcox_normalize(col_train, col_valid, col_test, repl_method='5-95perc'):
    
    col_train_norm, att = stats.boxcox(col_train)
    col_valid_norm = stats.boxcox(col_valid, att)
    col_test_norm = stats.boxcox(col_test, att)
    
    col_train_norm = pd.Series(col_train_norm)
    col_valid_norm = pd.Series(col_valid_norm)
    col_test_norm = pd.Series(col_test_norm)
    
    value1, value2 = get_values_for_replacement(col_train_norm, repl_method)
    l_whisker, r_whisker = get_whiskers(col_train_norm)
    
    col_train = col_train_norm.apply(replace_outliers, args=(l_whisker, r_whisker, value1, value2))
    col_valid = col_valid_norm.apply(replace_outliers, args=(l_whisker, r_whisker, value1, value2))
    col_test = col_test_norm.apply(replace_outliers, args=(l_whisker, r_whisker, value1, value2))

    return np.array(col_train), np.array(col_valid), np.array(col_test)


def normalize(func, col_train, col_valid, col_test, repl_method='5-95perc'):
    col_train_norm = pd.Series(func(col_train))
    col_valid_norm = pd.Series(func(col_valid))
    col_test_norm = pd.Series(func(col_test))
   
    value1, value2 = get_values_for_replacement(col_train_norm, repl_method)
    l_whisker, r_whisker = get_whiskers(col_train_norm)
    
    col_train = col_train_norm.apply(replace_outliers, args=(l_whisker, r_whisker, value1, value2))
    col_valid = col_valid_norm.apply(replace_outliers, args=(l_whisker, r_whisker, value1, value2))
    col_test = col_test_norm.apply(replace_outliers, args=(l_whisker, r_whisker, value1, value2))

    return np.array(col_train), np.array(col_valid), np.array(col_test)


def replace_bedrooms_number(old_value, new_value, df):
    df.loc[df['bedrooms']==old_value,'bedrooms'] = new_value
    
    return df


def remove_rows(df):
    df = df.drop(df[(df['bedrooms']==0)&(df['bathrooms']==0.0)].index)

    return df


def create_recon_age_col(df):
    df["date"] = df["date"].apply(lambda x: parse(x, dayfirst=False))
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    df['recon_age'] = df[['yr_renovated', 'yr_built', 'date']].apply(lambda x: calc_recon_age(x['yr_renovated'], x['yr_built'], x['date'].year), axis=1)
    
    return df['recon_age']
    

def run_pipeline(df):
    
    df = replace_bedrooms_number(33, 3, df)
    df = remove_rows(df)
    df['recon_age'] = create_recon_age_col(df)
    df = encode_zipcode(df)
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    
    #normalize price
    y_train, y_valid, y_test = boxcox_normalize(y_train, y_valid, y_test)
    
    #normalize sqft_living
    normalized = boxcox_normalize(X_train['sqft_living'], X_valid['sqft_living'], X_test['sqft_living'])
    X_train['sqft_living'] = normalized[0]
    X_valid['sqft_living'] = normalized[1]
    X_test['sqft_living'] = normalized[2]
    
    #normalize sqft_lot
    normalized = boxcox_normalize(X_train['sqft_lot'], X_valid['sqft_lot'], X_test['sqft_lot'], repl_method='med')
    X_train['sqft_lot'] = normalized[0]
    X_valid['sqft_lot'] = normalized[1]
    X_test['sqft_lot'] = normalized[2]
    
    
    #normalize sqft_above
    X_train['sqft_above'], X_valid['sqft_above'], X_test['sqft_above'] = boxcox_normalize(X_train['sqft_above'], X_valid['sqft_above'], X_test['sqft_above'])
    
    #normalize sqft_basement
    X_train['sqft_basement'], X_valid['sqft_basement'], X_test['sqft_basement'] = normalize(np.sqrt, X_train['sqft_basement'], X_valid['sqft_basement'], X_test['sqft_basement'], repl_method='mean')
    
    #normalize sqft_living15
    X_train['sqft_living15'], X_valid['sqft_living15'], X_test['sqft_living15'] = normalize(np.log, X_train['sqft_living15'], X_valid['sqft_living15'], X_test['sqft_living15'])
    
    #normalize sqft_lot15
    X_train['sqft_lot15'], X_valid['sqft_lot15'], X_test['sqft_lot15'] = boxcox_normalize(X_train['sqft_lot15'], X_valid['sqft_lot15'], X_test['sqft_lot15'], repl_method='med')
    
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test



