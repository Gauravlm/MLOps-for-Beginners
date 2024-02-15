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

    # data injestion
    df = data_injestion(data_path)
    # data_clean
    data_clean(df)
    # model train
    train_model(df)
    # evaluate model
    evaluation_model(df)
    logging.info(f'-----Train pipeline-----')