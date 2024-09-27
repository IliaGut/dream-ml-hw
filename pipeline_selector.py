from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd
from typing import Callable
from typing_extensions import Self


class PipelineSelector(BaseEstimator):
    def __init__(self, pipelines: dict, scorer: str, cv: int = 5, random_state: int = 0) -> None:
        """
        this class gets a dictionary of sklearn Pipelines, finds the best pipeline based on cross validation
        and uses it for predictions.

        :param pipelines: a dictionary where keys are pipeline names and values are sklearn Pipeline objects
        :param scoring: scoring used to select the best pipeline - sklearn.metrics.get_scorer_names()
        :param cv: number of cross-validation folds
        :param random_state: random state to ensure reproducability
        """

        self.pipelines = pipelines
        self.scorer = scorer
        self.cv = cv
        self.best_pipeline = None
        self.best_score  = -np.inf
        self.scores = {}
        self.random_state = random_state    

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        """
        fit all pipelines on the data and select the best one based on gridsearccv performance.
        
        :param X: training features
        :param y: training labels
        :return: self
        """

        for name, pipeline in self.pipelines.items():
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            kf.get_n_splits(X)
            scores = cross_val_score(pipeline, X, y, cv=kf, scoring=self.scorer)
            mean_score = np.mean(scores)
            self.scores[name] = mean_score
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_pipeline = pipeline
        
        if self.best_pipeline:
            self.best_pipeline.fit(X, y)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        use the best pipeline to make predictions.
        
        :param X: test features
        :return: predicted labels
        """

        if self.best_pipeline is None:
            raise ValueError("You must fit a model before making predictions.")

        return self.best_pipeline.predict(X)
