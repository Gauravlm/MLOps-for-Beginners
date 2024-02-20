import pandas as pd
from  abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import logging

class Model:
    '''
    Abstract Method for all model
    '''
    @abstractmethod
    def train(self, X_train,y_train):
        '''
        Args:
            X_train: Training data
            y_train: training label
        '''

        pass

class LinearRegressionModel(Model):
    '''
    Linear Regression Model
    '''
    def train(self, X_tain,y_train,**kwarg):
        '''
        Train the model
        Args:
            X_train: Training data
            y_train: training label
        '''
        try:
            reg= LinearRegression(**kwarg)
            reg.fit(X_tain,y_train)
            logging.info(f' ---- Traning Model is completed ----' )
            return reg
        except Exception as e:
            logging.error(f'Error in Training Model {e}')
            raise e
        

        
