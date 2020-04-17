import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def check_correlations(df, threshold):
    col_corr = set()
    corr_matrix = df.corr()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in df.columns:
                    del df[colname]

    return list(df.columns)


def feature_filter(X_train, y_train, threshold):
    """
    Step 1: select columns with correlation koeficient value which is above threshold.
    Step 2: select only those columns which do not correlate with each other (if so, choose only one)
    """
    df = X_train.copy()
    df['price'] = y_train
    correlation = df.corr()

    cor_target = abs(correlation['price'])
    relevant_features = cor_target[(cor_target > threshold) & (cor_target < 1.0)]
    features = relevant_features.index
    feaures = check_correlations(df, threshold+0.2)
    
    return features


def feature_wrapper(X_train, y_train):
    
    cols = list(X_train.columns)
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


def select_features_SFS(X_train, y_train, model_func, k = 10, forward = True, floating = False, scoring = 'r2', cv = 0):
    # selects features with Sequential Feature Selector
    sfs = SFS(model_func(), k_features=k, forward=forward, floating=floating, scoring = scoring, cv = cv)
    cols = list(X_train.columns)
    sfs.fit(X_train[cols], y_train)
    features = list(sfs.k_feature_names_)
    
    return features


def select_features_RFE(X_train, X_test, y_train, y_test):
    # selects features with Recursion Forward Elimination algorithm
    high_score=0
    nof=0           
    score_list =[]

    cols = list(X_train.columns)
    nof_list = np.arange(len(cols))

    for n in range(1,len(cols)):
        model = linear_model.LinearRegression()
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train[cols],np.ravel(y_train))
        X_test_rfe = rfe.transform(X_test[cols])

        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)

        if(score>high_score):
            high_score = score
            nof = nof_list[n]

    print("Optimum number of features: %d (all features: %d)" %(nof, len(cols)))
    print("Score with %d features: %f" % (nof, high_score))

    rfe = RFE(model, nof)
    X_rfe = rfe.fit_transform(X_train[cols],np.ravel(y_train))
    model.fit(X_rfe,y_train)              
    temp = pd.Series(rfe.support_,index = cols)
    selected_features = list(temp[temp==True].index)
    
    return selected_features