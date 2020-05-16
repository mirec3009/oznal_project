import random
import itertools
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor

from src import feature_selection2, metrics2

class Trees:
    def __init__(self, X_train, X_valid, y_train, y_valid, price_lambda):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.price_lambda = price_lambda


    def eval_model(self, model, iteration):
        model.fit(self.X_train, np.ravel(self.y_train))

        y_1 = model.predict(self.X_valid)
        y_2 = model.predict(self.X_train)
        
        mean_squared_error=metrics.mean_squared_error(inv_boxcox(self.y_valid, self.price_lambda),inv_boxcox(y_1, self.price_lambda))
        rmlse = metrics.mean_squared_log_error(inv_boxcox(self.y_valid, self.price_lambda), inv_boxcox(y_1, self.price_lambda))

        mse_train = metrics.mean_squared_error(inv_boxcox(self.y_train, self.price_lambda),inv_boxcox(y_2, self.price_lambda))
        rmlse_train = metrics.mean_squared_log_error(inv_boxcox(self.y_train, self.price_lambda), inv_boxcox(y_2, self.price_lambda))
        
        r2_train = model.score(self.X_train, self.y_train)
        r2_test = model.score(self.X_valid, self.y_valid)
        adj_r2_train = 1 - (((1 - r2_train) * (self.X_train.shape[0] - 1)) / (self.X_train.shape[0] - self.X_train.shape[1] - 1))
        adj_r2_test = 1 - (((1 - r2_test) * (self.X_valid.shape[0] - 1)) / (self.X_valid.shape[0] - self.X_valid.shape[1] - 1))
        
        score = {
            "MSE train": round(mse_train, 2),
            "RMSE train": round(np.sqrt(mse_train),2),
            "RMSLE train": round(np.sqrt(rmlse_train),2),
            "MSE": round(mean_squared_error, 2),
            "RMSE": round(np.sqrt(mean_squared_error),2),
            "RMSLE": round(np.sqrt(rmlse),2),
            "R2_train": round(r2_train,3),
            "R2_test": round(r2_test,3),
            "Adj_R2_train": round(adj_r2_train,3),
            "Adj_R2_test": round(adj_r2_test,3)
        }
        
        return score


    def grid_search(self, model_func,hyperparams, max_iterations):
        
        df_results = pd.DataFrame(columns = ['MSE train', 'RMSE train', 'RMSLE train', 'MSE', 'RMSE', 'RMSLE', 'R2_train', 'R2_test', 'Adj_R2_train', 'Adj_R2_test', 'params'],
                                      index = list(range(max_iterations)))
        
        keys, values = zip(*hyperparams.items())

        for i, v in enumerate(itertools.product(*values)):
            print(f"Iteration {i} in process...", end='')
            hyperparameters = dict(zip(keys, v))
            
            model = model_func(**hyperparameters)
            eval_results = self.eval_model(model, i)
            
            for k,v in eval_results.items():
                df_results[k][i] = v

            df_results['params'][i] = hyperparameters
            
            print("done")
            if i >= max_iterations-1:
                break
           
        df_results['delta_RMSE'] = abs(df_results["RMSE train"]-df_results["RMSE"])
        # Sort with best score on top
        df_results.sort_values(['RMSE', 'R2_test'], ascending = [True, False], inplace = True)
        df_results.reset_index(inplace=True)
        
        return df_results


    def random_search(self, model_func, hyperparams, max_iterations):
        
        df_results = pd.DataFrame(columns = ['MSE train', 'RMSE train', 'RMSLE train', 'MSE', 'RMSE', 'RMSLE', 'R2_train', 'R2_test', 'Adj_R2_train', 'Adj_R2_test', 'params'],
                                      index = list(range(max_iterations)))
        
        for i in range(max_iterations):
            print(f"Iteration {i} in process...", end='')
            hyperparameters = {k: random.choice(v) for k, v in hyperparams.items()}
            
            model = model_func(**hyperparameters)
            eval_results = self.eval_model(model, i)
            
            for k,v in eval_results.items():
                df_results[k][i] = v
            df_results['params'][i] = hyperparameters
            print("done")
            
        df_results['delta_RMSE'] = abs(df_results["RMSE train"]-df_results["RMSE"])
        # Sort with best score on top
        df_results.sort_values(['RMSE', 'R2_test'], ascending = [True, False], inplace = True)
        df_results.reset_index(inplace=True)
        
        return df_results

class SGD:
    def __init__(self, X_train, X_valid, y_train, y_valid, price_lambda):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.price_lambda = price_lambda


    def eval_model(self, clf, iteration, features1):

        feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=300)

        data_transformed_train = feature_map_nystroem.fit_transform(self.X_train[features1])
        data_transformed_valid = feature_map_nystroem.transform(self.X_valid[features1])

        clf.fit(data_transformed_train, np.ravel(self.y_train))
        y_hat = clf.predict(data_transformed_valid)
        
        mean_squared_error=metrics.mean_squared_error(inv_boxcox(self.y_valid, self.price_lambda),inv_boxcox(y_hat, self.price_lambda))
        rmlse = metrics.mean_squared_log_error(inv_boxcox(self.y_valid, self.price_lambda), inv_boxcox(y_hat, self.price_lambda))

        r2_train = clf.score(data_transformed_train, self.y_train)
        r2_test = clf.score(data_transformed_valid, self.y_valid)
        adj_r2_train = 1 - (((1 - r2_train) * (data_transformed_train.shape[0] - 1)) / (data_transformed_train.shape[0] - data_transformed_train.shape[1] - 1))
        adj_r2_test = 1 - (((1 - r2_test) * (data_transformed_valid.shape[0] - 1)) / (data_transformed_valid.shape[0] - data_transformed_valid.shape[1] - 1))
        
        score = {
            "MSE": round(mean_squared_error, 2),
            "RMSE": round(np.sqrt(mean_squared_error),2),
            "RMSLE": round(np.sqrt(rmlse),2),
            "R2_train": round(r2_train,3),
            "R2_test": round(r2_test,3),
            "Adj_R2_train": round(adj_r2_train,3),
            "Adj_R2_test": round(adj_r2_test,3)
        }
        
        return score


    def random_search(self, model_func, hyperparams, max_iterations, features):
        
        df_results = pd.DataFrame(columns = ['MSE', 'RMSE', 'RMSLE', 'R2_train', 'R2_test', 'Adj_R2_train', 'Adj_R2_test', 'params'],
                                      index = list(range(max_iterations)))
        
        for i in range(max_iterations):
            hyperparameters = {k: random.choice(v) for k, v in hyperparams.items()}
            model = model_func(**hyperparameters)
            eval_results = self.eval_model(model, i, features)
            for k,v in eval_results.items():
                df_results[k][i] = v
            df_results['params'][i] = hyperparameters
        
        df_results['delta_R2'] = abs(df_results["R2_train"]-df_results["R2_test"])
        # Sort with best score on top
        df_results.sort_values(['RMSE', 'R2_test'], ascending = [True, False], inplace = True)
        df_results.reset_index(inplace=True)
        
        return df_results


class LinearSVR:
    def __init__(self, X_train, X_valid, y_train, y_valid, price_lambda):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.price_lambda = price_lambda

    def eval_model(self, clf, iteration, features1):

        X_my_train = self.X_train[features1]#.values.reshape(-1,1)
        X_my_valid = self.X_valid[features1]#.values.reshape(-1,1)

        clf.fit(X_my_train, np.ravel(self.y_train))

        y_hat = clf.predict(X_my_valid)
        mean_squared_error=metrics.mean_squared_error(inv_boxcox(self.y_valid, self.price_lambda),inv_boxcox(y_hat, self.price_lambda))
        rmlse = metrics.mean_squared_log_error(inv_boxcox(self.y_valid, self.price_lambda), inv_boxcox(y_hat, self.price_lambda))

        r2_train = clf.score(X_my_train, self.y_train)
        r2_test = clf.score(X_my_valid, self.y_valid)
        adj_r2_train = 1 - (((1 - r2_train) * (X_my_train.shape[0] - 1)) / (X_my_train.shape[0] - X_my_train.shape[1] - 1))
        adj_r2_test = 1 - (((1 - r2_test) * (X_my_valid.shape[0] - 1)) / (X_my_valid.shape[0] - X_my_valid.shape[1] - 1))
        
        score = {
            "MSE": round(mean_squared_error, 2),
            "RMSE": round(np.sqrt(mean_squared_error),2),
            "RMSLE": round(np.sqrt(rmlse),2),
            "R2_train": round(r2_train,3),
            "R2_test": round(r2_test,3),
            "Adj_R2_train": round(adj_r2_train,3),
            "Adj_R2_test": round(adj_r2_test,3)
        }
        
        return score

    def random_search(self, model_func, hyperparams, max_iterations, features):
        
        df_results = pd.DataFrame(columns = ['MSE', 'RMSE', 'RMSLE', 'R2_train', 'R2_test', 'Adj_R2_train', 'Adj_R2_test', 'params'],
                                      index = list(range(max_iterations)))
        
        for i in range(max_iterations):
            hyperparameters = {k: random.choice(v) for k, v in hyperparams.items()}
            model = model_func(**hyperparameters)
            eval_results = self.eval_model(model, i, features)
            for k,v in eval_results.items():
                df_results[k][i] = v
            df_results['params'][i] = hyperparameters
        
        df_results['delta_R2'] = abs(df_results["R2_train"]-df_results["R2_test"])
        # Sort with best score on top
        df_results.sort_values(['RMSE', 'R2_test'], ascending = [True, False], inplace = True)
        df_results.reset_index(inplace=True)
        
        return df_results