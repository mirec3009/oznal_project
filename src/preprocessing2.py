import pandas as pd
import numpy as np
import category_encoders as ce
import scipy.stats as stats
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from src.analysis import get_whiskers, calc_recon_age
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def create_price_per_sqft_column(X_train, X_valid, X_test, y_train):
    X_train['price_per_sqft'] = y_train / (X_train['sqft_living'] + X_train['sqft_basement'])
    median = X_train['price_per_sqft'].median()
    X_train['price_per_sqft'] = X_train.groupby('zipcode')['price_per_sqft'].transform(lambda x: x.median())
    
    d = {}
    for _, x_train in X_train.iterrows():
        d.update({x_train['zipcode']:x_train['price_per_sqft']})
        
    X_valid['price_per_sqft'] = median
    for indice, x_valid in X_valid.iterrows():
        X_valid.loc[X_valid.index == indice ,'price_per_sqft'] = d[x_valid['zipcode']]
        
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
    
    X_train_encoded.drop(columns=['zipcode_0'])
    X_valid_encoded.drop(columns=['zipcode_0'])
    X_test_encoded.drop(columns=['zipcode_0'])
    
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


def create_recon_age_col(df):
    df["date"] = df["date"].apply(lambda x: parse(x, dayfirst=False))
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    df['recon_age'] = df[['yr_renovated', 'yr_built', 'date']].apply(lambda x: calc_recon_age(x['yr_renovated'], x['yr_built'], x['date'].year), axis=1)
    
    return df['recon_age']


def feature_filter(X_train, y_train, threshold):
    
    df = X_train.copy()
    df['price'] = y_train
    correlation = df.corr()

    cor_target = abs(correlation["price"])
    relevant_features = cor_target[(cor_target > threshold) & (cor_target < 1.0)]
    features = relevant_features.index
    
    return features


def feature_wrapper(X_train, y_train):
    
    cols = list(X_train.columns)
    cols.remove('date')
    pmax = 1
    while (len(cols)>0):

        X_1 = X_train[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y_train,X_1.astype(float)).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break

    return cols


def select_features(X_train, y_train, model_func, k = 10, forward = True, floating = False, scoring = 'r2', cv = 0):
    sfs = SFS(model_func(), k_features=k, forward=forward, floating=floating, scoring = scoring, cv = cv)
    cols = list(X_train.columns)
    cols.remove('date')
    sfs.fit(X_train[cols], y_train)
    features = list(sfs.k_feature_names_)
    
    return features

def run_pipeline(df):
    
    df = replace_bedrooms_number(33, 3, df)
    df = remove_rows(df)
    df['recon_age'] = create_recon_age_col(df)
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    
    #normalize price
    y_train, y_valid, y_test, price_lambda = boxcox_normalize(y_train, y_valid, y_test)
    
    #create price per sqft column
    X_train, X_valid, X_test = create_price_per_sqft_column(X_train, X_valid, X_test, y_train)
    
    X_train, X_valid, X_test = encode_zipcode(X_train, X_valid, X_test)
    
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
    normalized = boxcox_normalize(X_train['sqft_above'], X_valid['sqft_above'], X_test['sqft_above'])
    X_train['sqft_above'] = normalized[0]
    X_valid['sqft_above'] = normalized[1]
    X_test['sqft_above'] = normalized[2]
    
    #normalize sqft_basement
    X_train['sqft_basement'], X_valid['sqft_basement'], X_test['sqft_basement'] = normalize(np.sqrt, X_train['sqft_basement'], X_valid['sqft_basement'], X_test['sqft_basement'], repl_method='mean')
    
    #normalize sqft_living15
    X_train['sqft_living15'], X_valid['sqft_living15'], X_test['sqft_living15'] = normalize(np.log, X_train['sqft_living15'], X_valid['sqft_living15'], X_test['sqft_living15'])
    
    #normalize sqft_lot15
    normalized = boxcox_normalize(X_train['sqft_lot15'], X_valid['sqft_lot15'], X_test['sqft_lot15'], repl_method='med')
    X_train['sqft_lot15'] = normalized[0]
    X_valid['sqft_lot15'] = normalized[1]
    X_test['sqft_lot15'] = normalized[2]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, price_lambda



