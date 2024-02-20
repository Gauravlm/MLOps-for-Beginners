from sklearn.metrics import mean_squared_error,r2_score
from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np


class Evaluation:
    '''
    Abstact class define Evalutin stategy
    '''
    @abstractmethod
    def calculate_scroe(self, y_pred:np.ndarray, y_test: np.ndarray):
        '''
        Args:
            y_pred: Predicted label
            y_test:  Test lable
        Return:
            None
        '''
        pass

class MSE(Evaluation):
    def calculate_scroe(self, y_pred: np.ndarray, y_test: np.ndarray):
        try:
            logging.info('Calculating MSE')
            mse= mean_squared_error(y_pred, y_test)
            logging.info(f'MSE:- {mse}')
            return mse
        except Exception as e:
            logging.error(f'Error in calculating MSE: {e}')
            raise e
        
class R2_score(Evaluation):
    def calculate_scroe(self, y_pred: np.ndarray, y_test: np.ndarray):
        try:
            logging.info('Calculating R2 Score')
            r2= r2_score(y_pred,y_test)
            logging.info(f'R2-score: {r2}')
            return r2
        except Exception as e:
            logging.error(f'Error in calculating R2-score {e}')
            raise e

class RMSE(Evaluation):
    '''
    Calcualte Root Mean Squared Error
    '''
    def calculate_scroe(self, y_pred: np.ndarray, y_test: np.ndarray):
        try:
            logging.info('Calculating RMSE')
            rmse= mean_squared_error(y_pred,y_test, squared=False)
            logging.info(f'RMSE: {rmse}')
            return rmse
        
        except Exception as e:
            logging.error(f'Error in calculating RMSE: {e}')
            raise e