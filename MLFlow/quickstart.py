import os
import tempfile
from pathlib import Path
from random import random, randint

import mlflow

with mlflow.start_run():
    # Log a parameter (key-value pair)
    mlflow.log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    mlflow.log_metric(key="foo", value=random())
    mlflow.log_metric(key="foo", value=random() + 1)
    mlflow.log_metric(key="foo", value=random() + 2)

    # Log an artifact (output file)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        with (tmp_dir / "test.txt").open("w") as f:
            f.write("hello world!")
        mlflow.log_artifacts(tmp_dir, artifact_path="output")
