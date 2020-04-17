import pandas as pd
import numpy as np
import category_encoders as ce
import scipy.stats as stats
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from src.analysis import get_whiskers, calc_recon_age


def create_price_per_sqft_column(X_train, X_valid, X_test, y_train):
    #fit_transform part on train
    X_train['price_per_sqft'] = y_train / (X_train['sqft_living'] + X_train['sqft_basement'])
    # default median for missing zipcode values in valid/test
    median = X_train['price_per_sqft'].median()
    #for each zipcode get median of price_per_sqft
    X_train['price_per_sqft'] = X_train.groupby('zipcode')['price_per_sqft'].transform(lambda x: x.median())
    
    # transform part on vald/test
    #creates dictionary with median for each zipcode value in train
    d = {}
    for _, x_train in X_train.iterrows():
        d.update({x_train['zipcode']:x_train['price_per_sqft']})
    
    #based on train dictionary, choose correct price_per_sqft (determined by zipcode value), new zipcode values has default median
    X_valid['price_per_sqft'] = median
    for indice, x_valid in X_valid.iterrows():
        X_valid.loc[X_valid.index == indice ,'price_per_sqft'] = d[x_valid['zipcode']]
    
    #based on train dictionary, choose correct price_per_sqft (determined by zipcode value), new zipcode values has default median
    X_test['price_per_sqft'] = median
    for indice, x_test in X_test.iterrows():
        X_test.loc[X_test.index == indice ,'price_per_sqft'] = d[x_test['zipcode']]    
        
    return X_train, X_valid, X_test


def split_data(df):
    df_sorted = df.sort_values(by='date')
    X = df_sorted.loc[:, df_sorted.columns != 'price']
    y = df_sorted['price']

    #split 70:20:10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.33, shuffle=False)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def encode_zipcode(X_train, X_valid, X_test):
    encoder = ce.binary.BinaryEncoder(cols=['zipcode'])
    X_train_encoded = encoder.fit_transform(X_train)
    X_valid_encoded = encoder.transform(X_valid)
    X_test_encoded = encoder.transform(X_test)
    
    X_train_encoded.drop(columns=['zipcode_0'], inplace=True)
    X_valid_encoded.drop(columns=['zipcode_0'], inplace=True)
    X_test_encoded.drop(columns=['zipcode_0'], inplace=True)
    
    return X_train_encoded, X_valid_encoded, X_test_encoded


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

    return np.array(col_train), np.array(col_valid), np.array(col_test), att


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


def remove_col(df, col_list):
    df = df.drop(columns=col_list)
    
    return df


def create_recon_age_col(df):
    df["date"] = df["date"].apply(lambda x: parse(x, dayfirst=False))
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    df['recon_age'] = df[['yr_renovated', 'yr_built', 'date']].apply(lambda x: calc_recon_age(x['yr_renovated'], x['yr_built'], x['date'].year), axis=1)
    
    return df['recon_age']


def run_pipeline(df):
    # replace outlying number of bedrooms (33)
    df = replace_bedrooms_number(33, 3, df)
    
    #remove rows which have 0 bedrooms and 0.0 bathrooms
    df = remove_rows(df)
    
    #create new column recon_age, which describes number of years from last reconstruction till year of sale(if no reconstruction was made, then the number of years from year of building to year of sale)
    df['recon_age'] = create_recon_age_col(df)
    
    # sort data by date of sale, then split in ratio 70:20:10
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    
    #remove date column
    X_train = remove_col(X_train, ['date'])
    X_valid = remove_col(X_valid, ['date'])
    X_test = remove_col(X_test, ['date'])
    
    #normalize price with boxcox, replace outliers with 5-95 percentile
    y_train, y_valid, y_test, price_lambda = boxcox_normalize(y_train, y_valid, y_test)
    
    #create new column price_per_sqft, which describes ratio of price column and sum of sqft_living with sqft_basement
    X_train, X_valid, X_test = create_price_per_sqft_column(X_train, X_valid, X_test, y_train)
    
    #binary encoding for zipcode column
    X_train, X_valid, X_test = encode_zipcode(X_train, X_valid, X_test)
    
    #normalize sqft_living with boxcox, replace outliers with 5-95 percentile
    normalized = boxcox_normalize(X_train['sqft_living'], X_valid['sqft_living'], X_test['sqft_living'])
    X_train['sqft_living'] = normalized[0]
    X_valid['sqft_living'] = normalized[1]
    X_test['sqft_living'] = normalized[2]
    
    #normalize sqft_lot with boxcox, replace outliers with median
    normalized = boxcox_normalize(X_train['sqft_lot'], X_valid['sqft_lot'], X_test['sqft_lot'], repl_method='med')
    X_train['sqft_lot'] = normalized[0]
    X_valid['sqft_lot'] = normalized[1]
    X_test['sqft_lot'] = normalized[2]
    
    #normalize sqft_above with boxcox, replace outliers with 5-95 percentile
    normalized = boxcox_normalize(X_train['sqft_above'], X_valid['sqft_above'], X_test['sqft_above'])
    X_train['sqft_above'] = normalized[0]
    X_valid['sqft_above'] = normalized[1]
    X_test['sqft_above'] = normalized[2]
    
    #normalize sqft_basement with boxcox, replace outliers with mean
    normalized = normalize(np.sqrt, X_train['sqft_basement'], X_valid['sqft_basement'], X_test['sqft_basement'], repl_method='mean')
    X_train['sqft_basement'] = normalized[0]
    X_valid['sqft_basement'] = normalized[1]
    X_test['sqft_basement'] = normalized[2]
    
    #normalize sqft_living15 with log naturalis, replace with 5-95 percentile
    normalized = normalize(np.log, X_train['sqft_living15'], X_valid['sqft_living15'], X_test['sqft_living15'])
    X_train['sqft_living15'] = normalized[0]
    X_valid['sqft_living15'] = normalized[1]
    X_test['sqft_living15'] = normalized[2]
    
    #normalize sqft_lot15 with boxcox, replace outliers with median
    normalized = boxcox_normalize(X_train['sqft_lot15'], X_valid['sqft_lot15'], X_test['sqft_lot15'], repl_method='med')
    X_train['sqft_lot15'] = normalized[0]
    X_valid['sqft_lot15'] = normalized[1]
    X_test['sqft_lot15'] = normalized[2]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, price_lambda



