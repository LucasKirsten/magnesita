""" Script to handle the data flow """

import os
from glob import glob

import pickle as pkl

from math import erfc
import pandas as pd
import numpy as np

import src.data_utils as du

class DataLoader(object):
    def __init__(self, mode:str, path_save, path_raw:str, path_target:str=None, split_train_test:float=0):
        """
        Initializer method for the data loader.
        
        Parameters
        ----------
        mode : string
            Mode to be used when building the data handeling pipeline. Should be either train or test.
        path_save : string
            Path to save or load data handeling options.
        path_raw : string
            Path to the csv file containing the raw data.
        path_target : string (optional, default=None)
            Path to the csv file containing the target data.
        split_train_test : float (optional, default=0.1)
            Ratio to split data between training and testing.
        """
        
        assert mode in ('train', 'test'), 'Mode should be either train or test.'
        
        self.path_save = path_save
        
        # load used columns and normalization during training
        if mode=='test':
            columns = pkl.load(open(os.path.join(self.path_save, 'columns.pkl'), 'rb'))
            norm = pkl.load(open(os.path.join(self.path_save, 'norm.pkl'), 'rb'))
            self.norm_target = norm[0]
        else:
            columns = None
            norm = None
        
        # read csv's
        df_raw = pd.read_csv(path_raw, sep=';')
        if path_target:
            df_target = pd.read_csv(path_target, sep=';')
        else:
            df_target = None
        
        # make data handeling pipeline
        self.df = self.__make_data_pipeline(mode, df_raw, df_target, columns, norm, split_train_test)
            
    def get_data(self):
        """
        Return the input and output datas as numpy arrays
            
        Returns
        ----------
        X : numpy.array
            Input data.
        y : numpy.array
            Target data.
        """
        
        # verify if target is provided
        if 'target' in self.df.columns:
            X = np.array(self.df.drop('target', axis=1))
            y = np.array(self.df['target'])
        else:
            X = np.array(self.df)
            y = None
        
        return X,y
    
    def pred_denorm(self, x):
        """
        Denorm the predicted data.
        """
        
        return (x+1.)*(self.norm_target[0]-self.norm_target[1])/2.+self.norm_target[1]
    
    def __make_data_pipeline(self, mode, df_raw, df_target=None, columns=None, norm=None, ratio=0):
        """
        Make the train pipeline to adjust the training data
        """
        
        print('Making data pipeline...')
        
        # split between train and test
        df_raw_train, df_raw_test, df_target_train, df_target_test = du.split_train_test(df_raw, df_target, ratio)
        
        # save train and test in a csv file
        if mode=='train':
            df_raw_train.to_csv(os.path.join(self.path_save, 'train_raw.csv'), index=False, sep=';')
            df_target_train.to_csv(os.path.join(self.path_save, 'train_target.csv'), index=False, sep=';')

            df_raw_test.to_csv(os.path.join(self.path_save, 'test_raw.csv'), index=False, sep=';')
            df_target_test.to_csv(os.path.join(self.path_save, 'test_target.csv'), index=False, sep=';')

        # get the correct split for the pipeline
        if mode=='train':
            df_raw = df_raw_train.copy()
            df_target = df_target_train.copy()
        else:
            df_raw = df_raw_test.copy()
            if df_target is not None:
                df_target = df_target_test.copy()
        
        df_raw = du.adjust_dates(df_raw, du.dates)
        df_raw = du.remove_non_numerical(df_raw)
        
        # use only the columns used during training
        if columns is not None:
            df_raw = df_raw.loc[:, columns]
            
        if df_target is not None:
            df = du.merge_raw_target(df_raw, df_target)
        else:
            df = df_raw.copy()
        
        if mode=='train':
            df = du.fill_nan(df, mode='repeat')
        else:
            df = du.fill_nan(df, mode='mean')
        
        if mode=='train':
            df = du.remove_outliers(df)
            df = du.filter_constant(df)
        
        if mode=='train':
            df = du.fill_nan(df, mode='repeat')
        else:
            df = du.fill_nan(df, mode='mean', mean=norm[1][0])
        
        if mode=='train':
            df = du.remove_low_corr(df)
        
        df, norm = du.normalize_data(df, norm)
        
        # save the used columns and normalization during training
        if mode=='train':
            self.norm_target = norm[0]
            columns = df.drop('target', axis=1).columns
            pkl.dump(columns, open(os.path.join(self.path_save, 'columns.pkl'), 'wb'))
            pkl.dump(norm, open(os.path.join(self.path_save, 'norm.pkl'), 'wb'))
        
        return df