from pipelines.train_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path='/Users/gm/study/mlops/MLOps-for-Beginners/data/olist_customers_dataset.csv')

# mlflow ui --backend-store-uri "file:/Users/gm/Library/Application Support/zenml/local_stores/b7c5cea8-826d-400e-9121-9d0f959c97e4/mlruns"