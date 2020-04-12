import pandas as pd
import numpy as np
from scipy.special import inv_boxcox
from sklearn import metrics

def evaluate(model, X_train, y_train, X_test, y_test, price_lambda):
    mean_squared_error=metrics.mean_squared_error(inv_boxcox(y_test, price_lambda),inv_boxcox(pred, price_lambda))
    rmlse = metrics.mean_squared_log_error(inv_boxcox(y_test, price_lambda), inv_boxcox(pred, price_lambda))

    r2_train = reg.score(x_train,y_train)
    r2_test = reg.score(x_test,y_test)
    adj_r2_train = 1 - (((1 - r2_train) * (x_train.shape[0] - 1)) / (x_train.shape[0] - x_train.shape[1] - 1))
    adj_r2_test = 1 - (((1 - r2_test) * (x_test.shape[0] - 1)) / (x_test.shape[0] - x_test.shape[1] - 1))

    print('Mean Squared Error', round(mean_squared_error, 5))
    print('Root Mean Squared Error', round(np.sqrt(mean_squared_error),5))
    print('Root Mean Squared Log Error', round(np.sqrt(rmlse),5))
    print('R squared training',round(r2_train,3))
    print('R squared testing',round(r2_test,3))
    print('Adjusted-R squared training', round(adj_r2_train,3))
    print('Adjusted-R squared testing', round(adj_r2_test,3))
    print('intercept',reg.intercept_)
    print('coefficient',reg.coef_)