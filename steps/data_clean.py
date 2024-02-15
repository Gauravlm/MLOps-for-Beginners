from zenml import step
import pandas as pd
import logging

@step()
def data_clean(df:pd.DataFrame):
    logging.info(f'-----Data cleaning proecess-----')