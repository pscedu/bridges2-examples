import os
from random import random, randint
import mlflow

with mlflow.start_run():
    # Log a parameter (key-value pair)
    mlflow.log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    for epoch in range(0, 100):
        mlflow.log_metric(key="foo", value=random(), step=epoch)
