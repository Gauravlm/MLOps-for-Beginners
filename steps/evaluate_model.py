from zenml import step
import pandas as pd
import logging

@step()
def evaluation_model(df:pd.DataFrame):
    logging.info(f'-----Evaluate model-----')