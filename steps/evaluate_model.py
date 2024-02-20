from zenml import step
import pandas as pd
import logging
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2_score,RMSE
from typing import Tuple
from typing_extensions import Annotated

@step()
def evaluation_model( model:RegressorMixin,  X_test:pd.DataFrame, y_test: pd.Series):
    
    '''
    Args:
        X_test, y_test
    Return:
        r2_score, rmse
    '''
    logging.info(f'-----Evaluate model-----')
    try:
        y_pred= model.predict(X_test)
        mse = MSE()
        mse_score = mse.calculate_scroe(y_pred, y_test)

        r2 = R2_score()
        r2_sc = r2.calculate_scroe(y_pred,y_test)

        rmse = RMSE()
        rmse_score = rmse.calculate_scroe(y_pred,y_test)
        return [r2_sc , rmse_score]
        
    
    except Exception as e:
        logging.error(f'Error in Evaluation model {e}')
        raise e

    