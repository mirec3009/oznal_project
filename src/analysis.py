import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

def display_boxplot(column):
    ax = sns.boxplot(column)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()
    
    
def display_normalized(column, func):
    normalised_liv = func(column)
    pd.Series(normalised_liv).hist(bins=50)
    print("Skewness (asymetria): %.3f" %stats.skew(normalised_liv))

def get_whiskers(column):
    descr = column.describe()
    IQR = descr['75%'] - descr['25%']
    whisker_r = np.min([descr['max'], descr['75%'] + (1.5 * IQR)])
    whisker_l = np.max([descr['min'], descr['25%'] - (1.5 * IQR)])
    
    return whisker_l, whisker_r

def calc_recon_age(yr_renovated, yr_built, yr_sold):
    if yr_renovated > 0:
        return yr_sold - yr_renovated
    else:
        return yr_sold - yr_built