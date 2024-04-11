import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from materializer.custome.materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)

from zenml.integrations.mlflow.services import mlflow_model_developer_step
from zenml.steps import BaseParameters, Output

from steps.data_clean import data_clean
from steps.data_injestion import data_injestion
from steps.train_model import train_model
from steps.evaluate_model import evaluation_model

# docker setting
docker_settings = DockerSettings(required_integrations=[MLFLOW])
@pipeline(enable_cache=True, settings={'docker_settings': docker_settings})
def countinuous_deployment_pipeline(
    min_accuracy: float= 0.92,
    workers: int= 1,
    # if program is in loop then how much time it should stop the run --> DEFAULT_SERVICE_START_STOP_TIMEOUT
    timeout: int= DEFAULT_SERVICE_START_STOP_TIMEOUT 
):
    df= data_injestion()

                                 