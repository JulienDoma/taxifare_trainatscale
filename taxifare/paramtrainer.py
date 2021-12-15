
from taxifare.data import get_data, clean_df, holdout
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
from taxifare.mlflow import MLFlowBase

from sklearn.model_selection import GridSearchCV

import joblib


class ParamTrainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            "[FR] [Paris] [gmanchon] taxifare + 1",
            "https://mlflow.lewagon.co")

    def train(self, params):

        # results
        models = {}

        # iterate on models
        for model_name, model_params in params.items():

            line_count = model_params["line_count"]
            hyper_params = model_params["hyper_params"]

            # create a mlflow training
            self.mlflow_create_run()

            # log params
            self.mlflow_log_param("model_name", model_name)
            self.mlflow_log_param("line_count", line_count)
            for key, value in hyper_params.items():
                self.mlflow_log_param(key, value)

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

            # create gridsearch object
            grid_search = GridSearchCV(
                pipeline,
                param_grid=hyper_params,
                cv=5)

            # train with gridsearch
            grid_search.fit(X_train, y_train)

            # score gridsearch
            score = grid_search.score(X_test, y_test)

            # save the trained model
            joblib.dump(pipeline, f"{model_name}.joblib")

            # push metrics to mlflow
            self.mlflow_log_metric("score", score)

            # return the gridsearch in order to identify the best estimators and params
            models[model_name] = grid_search

        return models
