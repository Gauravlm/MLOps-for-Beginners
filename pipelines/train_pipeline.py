'''
Training Pipeline
 steps
    # data injestion
    # data_clean
    # model train
    # evaluate model

'''

from zenml import pipeline
from steps.data_injestion import data_injestion
from steps.data_clean import data_clean
from steps.train_model import train_model
from steps.evaluate_model import evaluation_model
import logging

@pipeline()
def train_pipeline(data_path:str):
    logging.info(f'-----Train pipeline-----')
    # data injestion
    df = data_injestion(data_path)
    # data_clean
    X_train,X_test, y_train, y_test  =data_clean(df)
    # model train
    model = train_model(X_train,X_test, y_train, y_test)
    # evaluate model
    model_score = evaluation_model(model, X_test,y_test )
    print(model_score)
    # r2_score, rmse = model_score
    # logging.info(f'Model score :- {model_score}')
    # logging.info(f'Model score: {r2_score}')
    