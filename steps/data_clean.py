from zenml import step
import pandas as pd
import logging
from src.data_cleaning import DataCleaning, DataDivideStrategy,DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step()
def data_clean(df:pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame,'X_train'],
    Annotated[pd.DataFrame,'X_test'],
    Annotated[pd.Series,'y_train'],
    Annotated[pd.Series,'y_test']
]:
    '''
    Clean the data and split into train ,test
    Args:
        df: dataframe/raw data
    Return:
        X_train: Training Data
        X_test: Testing Data
        y_train: Training Label
        y_test: Testing Lable
    '''
    logging.info(f'-----Data cleaning proecess-----')
    try:
        # data processing part
        process_strategy= DataPreprocessStrategy()
        data_clean = DataCleaning(df, process_strategy)
        clean_data = data_clean.handle_data() # remove the columns
        logging.info('---Processing data is completed')

        # train_test_split
        divide_strategy = DataDivideStrategy()
        split_data = DataCleaning(clean_data,divide_strategy)
        X_train, X_test, y_train, y_test= split_data.handle_data() # split the data into train,test
        logging.info('----Spliting data into train-test is completed---')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error('---- Error in Data Cleaning ----')
        raise e
    

