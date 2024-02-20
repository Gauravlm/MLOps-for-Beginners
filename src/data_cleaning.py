import pandas as pd
from abc import ABC, abstractmethod # https://www.geeksforgeeks.org/abstract-base-class-abc-in-python/
from typing import Union
import numpy as np
from sklearn.model_selection import train_test_split
import logging 


class DataStrategy(ABC):
    '''
    Abstract class defining strategy for handling data 
    '''
    @abstractmethod
    def handle_data(self,df:pd.DataFrame)-> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    '''
    Data preprocessing strategy 
    '''
    def handle_data(self,df:pd.DataFrame) -> pd.DataFrame:
        '''
        - Remove the unwanted columns
        - Fill Null values with median
        - convert dtype of the columns

        '''

        try:
            df = df.drop([
                "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
            ], axis=1)

            df["product_weight_g"].fillna(df["product_weight_g"].median(), inplace=True)
            df["product_length_cm"].fillna(df["product_length_cm"].median(), inplace=True)
            df["product_height_cm"].fillna(df["product_height_cm"].median(), inplace=True)
            df["product_width_cm"].fillna(df["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            df["review_comment_message"].fillna("No review", inplace=True)

            df = df.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            df = df.drop(cols_to_drop, axis=1)

            return df
        
        except Exception as e:
            logging.error(e)
            raise e

class DataDivideStrategy(DataStrategy):
    '''
    split the data into  train test
    '''
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        '''
        devide the data into test and train
        '''
        try:
            X= df.drop('review_score', axis=1)
            y= df['review_score']
            X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        
class DataCleaning:
    '''
    Data cleaning class which process data and split into train , test
    '''
    def __init__(self, df:pd.DataFrame, strategy:DataStrategy) -> None:
        # initialize the class with specific stretegy
        self.df= df
        self.strategy= strategy

    def handle_data(self)-> Union[pd.DataFrame, pd.Series]:
        ''' Handle the data based on provided strategy '''
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error(e)
            raise e
        
   
        
