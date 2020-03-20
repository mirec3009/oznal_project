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
