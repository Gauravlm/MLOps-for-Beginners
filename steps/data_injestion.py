import pandas as pd
import logging
from  zenml import step

class DataInjestion:
    def __init__(self, datapath:str) :
        self.datapath= datapath

    def get_data(self):
        logging.info(f'Data Injestion from {self.datapath}')
        return pd.read_csv(self.datapath)


@step
def data_injestion(data_path:str) -> pd.DataFrame:
    '''
    Injesting data from data_paht
    Args:
        data_path: data path
    return:
        pd.DataFrame: injested data
    '''

    try:
        injestion = DataInjestion(data_path)
        df= injestion.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while injesting data: {e}')
        raise e
    