import pandas as pd
from zenml import step
import logging
from sklearn.base  import RegressorMixin
from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig
from zenml.client import Client
import mlflow

experiment_tracker= Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config:ModelNameConfig
)-> RegressorMixin:

    logging.info(f'-----Training model-----')
    model = None
    try:
        if config.model_name=='LinearRegression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            train_model = model.train(X_train,y_train)
            
            return train_model
        else :
            raise ValueError(f'Model {config.model_name} is not supported')
        
    except Exception as e:
        logging.error(f'Error in Train Model for {config.model_name}')
        raise e
    
        