
from taxifare.data import get_data, clean_df, holdout
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
from taxifare.metrics import compute_rmse
from taxifare.mlflow import MLFlowBase

import joblib


class Trainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            "[FR] [Paris] [gmanchon] taxifare + 1",
            "https://mlflow.lewagon.co")

    def train(self):

        model_name = "random_forest"
        line_count = 1_000

        # create a mlflow training
        self.mlflow_create_run()

        # log params
        self.mlflow_log_param("model_name", model_name)
        self.mlflow_log_param("line_count", line_count)

        # get data
        df = get_data(line_count)
        df = clean_df(df)

        # holdout
        X_train, X_test, y_train, y_test = holdout(df)

        # log params
        self.mlflow_log_param("model", model_name)

        # create model
        model = get_model(model_name)

        # create pipeline
        pipeline = get_pipeline(model)

        # train
        pipeline.fit(X_train, y_train)

        # make prediction for metrics
        y_pred = pipeline.predict(X_test)

        # evaluate metrics
        rmse = compute_rmse(y_pred, y_test)

        # save the trained model
        joblib.dump(pipeline, "model.joblib")

        # push metrics to mlflow
        self.mlflow_log_metric("rmse", rmse)

        # return the gridsearch in order to identify the best estimators and params
        return pipeline
