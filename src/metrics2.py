import pandas as pd
import numpy as np
from scipy.special import inv_boxcox
from sklearn import metrics

def evaluate(model, x_train, y_train, x_test, y_test, pred, price_lambda): 
    y_test_inv = inv_boxcox(y_test, price_lambda)
    pred_inv = inv_boxcox(pred, price_lambda)               
        
    mean_squared_error=metrics.mean_squared_error(y_test_inv,pred_inv)
    rmlse = metrics.mean_squared_log_error(y_test_inv, pred_inv)

    r2_train = model.score(x_train,y_train)
    r2_test = model.score(x_test,y_test)
    adj_r2_train = 1 - (((1 - r2_train) * (x_train.shape[0] - 1)) / (x_train.shape[0] - x_train.shape[1] - 1))
    adj_r2_test = 1 - (((1 - r2_test) * (x_test.shape[0] - 1)) / (x_test.shape[0] - x_test.shape[1] - 1))

    print('Mean Squared Error', f'{round(mean_squared_error, 5):,}')
    print('Root Mean Squared Error', f'{round(np.sqrt(mean_squared_error),5):,}')
    print('Root Mean Squared Log Error', f'{round(np.sqrt(rmlse),5):,}')
    print('R squared training',f'{round(r2_train,3):,}')
    print('R squared testing',f'{round(r2_test,3):,}')
    print('Adjusted-R squared training', f'{round(adj_r2_train,3):,}')
    print('Adjusted-R squared testing', f'{round(adj_r2_test,3):,}')
    print('intercept',model.intercept_)
    print('coefficient',model.coef_)