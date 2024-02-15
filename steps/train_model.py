import pandas as pd
from zenml import step
import logging
@step()
def train_model(df:pd.DataFrame):

    logging.info(f'-----Training model-----')