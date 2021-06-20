""" Script with the utils function for handling the data """

import os
from glob import glob

from math import erfc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# non numerical and dates columns
non_numerical = ['super_hero_group', 'crystal_type', 'Cycle']
dates = ['when', 'expected_start', 'start_process', 'start_subprocess1', 'start_critical_subprocess1', 'predicted_process_end', \
         'process_end', 'subprocess1_end', 'reported_on_tower', 'opened']

def merge_raw_target(df_raw:pd.DataFrame, df_target:pd.DataFrame):
    """
    Merge target DataFrame to the raw.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Data Frame containing the raw input data.
    df_target : pd.DataFrame
        Data Frame containing the target data.
        
    Returns
    ----------
    Data Frame
    """
    
    # merge the targets to the features
    df = df_raw.join(df_target['target'], lsuffix='_caller', rsuffix='_other')
    
    # get only data with target values
    df = df[:len(df_target)]
    
    return df
    
def adjust_dates(df:pd.DataFrame, dates:list):
    """
    Adjust the columns date to be used in the model training.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
    dates : list
        List of date columns in the df Data Frame.
        
    Returns
    ----------
    Data Frame
    """
    
    # convert string dates to pandas date and join hour, minute, seconds in a different column
    for col in dates:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
        df[col+'_time']   = df[col].dt.hour*60**2 + df[col].dt.minute*60 + df[col].dt.second
    
    return df

def adjust_non_numerical(df:pd.DataFrame, non_numerical:list):
    """
    Adjust the columns with non numerical values to be used in the model training.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
    non_numerical : list
        List of non numerical columns in the df Data Frame.
        
    Returns
    ----------
    Data Frame
    """
    
    # handle non numerical data by assignin unique numerical values
    for col in non_numerical:
        uniques = df[col].unique()
        mapping = dict(zip(uniques, range(len(uniques))))

        df[col] = df[col].apply(lambda x:mapping[x])
        
    return df

def remove_non_numerical(df:pd.DataFrame):
    """
    Remove columns with non numerical values.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
        
    Returns
    ----------
    Data Frame
    """
    
    # remove not numerical columns
    df = df._get_numeric_data()
    
    return df

def fill_nan(df:pd.DataFrame, mode:str='repeat', mean=None):
    """
    Fill NaN values on the Data Frame.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
    mode : string
        Mode to fill the Nan values. Should be either repeat or mean.
    mean : list
        List of mean values to be applied on the mean mode fill.
        
    Returns
    ----------
    Data Frame
    """
    
    assert mode in ('repeat', 'mean'), 'Mode should be either repeat or mean.'
    
    if mode=='repeat':
        # fill nan values with the previous value and then after value
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
    else:
        # fill with the mean value for each column
        if mean is not None:
            df = df.fillna(mean)
        else:
            df = df.fillna(df.mean())
    
    return df

def _chauvenet(array:np.array, s:int=3):
    """
    Evaluate chauvenet criterion on the index of the input array.

    Parameters
    ----------
    array : numpy.array
        Input data array
    s : int
        Number of standart deviations to be considered
        
    Returns
    ----------
    List of indexes with probability lower than criterion
    """
    
    N = len(array)
    criterion = 1.0/(s*N)
    norm = abs(array-array.mean())/array.std()
    prob = np.array([erfc(d) for d in norm])

    return prob < criterion

def remove_outliers(df:pd.DataFrame):
    """
    Remove outliers values using the Chauvenet Criterion
    and replace them by the value before or after it.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
        
    Returns
    ----------
    Data Frame
    """
    
    # use chauvenet criterion to eliminate outliners
    for col in df.columns:
        if col=='target':
            continue

        if len(df.loc[_chauvenet(df[col]), col])>0:
            df.loc[_chauvenet(df[col]), col] = np.nan
            
    return df

def filter_constant(df:pd.DataFrame, threshold:float=0.01):
    """
    Remove columns with standart deviation lower than the threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
    threshold : float
        Threshold to be used in the standart deviation filtering
        
    Returns
    ----------
    Data Frame
    """
    
    # filters by quasi-constant variance threshold
    constant_filter = VarianceThreshold(threshold=threshold)
    constant_filter.fit(df)
    df = df[df.columns[constant_filter.get_support()]]
    
    return df

def remove_low_corr(df:pd.DataFrame, threshold:float=0.01):
    """
    Remove columns with low correlation with the target column

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
    threshold : float
        Threshold to be used in the correlation filtering.
        
    Returns
    ----------
    Data Frame
    """
    
    # remove low correlated features with the targets
    correlated_features = set()
    correlation_matrix = df.corr()
    df = df.copy()

    df = df[df.columns[correlation_matrix['target'].abs() > threshold]]
    
    return df

def normalize_data(df:pd.DataFrame, norm=None):
    """
    Normalize the Data Frame. Input columns are normalized regarding their mean and standart deviation
    to have values mean=0 and std=1, while the Target column is normalized between the interval [-1,1]

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
    norm : tuple<tuple>
        Normalization escheme to be applied on the data.
        
    Returns
    ----------
    Data Frame
    """
    
    # normalize data
    df_norm = df.copy()
    
    # apply the normalization escheme
    if norm is not None:
        if 'target' not in df_norm.columns:
            df_norm = (df_norm-norm[1][0])/norm[1][1]
        else:
            df_norm = (df_norm.drop('target', axis=1)-norm[1][0])/norm[1][1]
            df_norm['target'] = 2*(df['target']-norm[0][1])/(norm[0][0]-norm[0][1])-1
    
    # define a normalization scheme
    else:
        for col in df_norm.columns:
            # output to [-1,1]
            if col=='target':
                df_norm[col] = 2*(df[col]-df[col].min())/(df[col].max()-df[col].min())-1
            # inputs to mean=0 and std=1
            else:
                df_norm[col] = (df[col]-df[col].mean())/df[col].std()
    
    # return values to use later in testing
    if norm is not None:
        return df_norm, norm
    elif 'target' in df_norm.columns:
        return df_norm, ((df['target'].max(), df['target'].min()), (df.drop('target', axis=1).mean(), df.drop('target', axis=1).std()))
    else:
        return df_norm, (None, (df.mean(), df.std()))
    
def split_train_test(df_raw, df_target, ratio):
    """
    Split data frame into training and testing.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame containing the data.
    ratio : float
        Ratio to split the data.
        
    Returns
    ----------
    Data Frame
    """
    
    assert ratio>=0 and ratio<1, 'Ratio for spliting the data should be between [0,1)'
    
    if ratio==0:
        return df_raw[:len(df_target)], df_raw[:len(df_target)], df_target, df_target
    return train_test_split(df_raw[:len(df_target)], df_target, test_size=ratio)