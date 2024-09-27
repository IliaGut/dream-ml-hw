from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Dict

from utils import load_class


class PipelineBuilder(object):

    def __init__(self, pipelines_cfg: dict = {}):
        """
        this class gets a configuration dictionary that is used to build an sklearn Pipeline object
        each key is the name of the pipeline and the values are a list where each element is a dict with
        the following keys:
            name - pipeline step name
            class - a transformer / estimator class applied in the step
            params - parameters to pass to the class
            cols (optional) - list of columns to apply the transformation on

        :param pipelines_cfg: dictionary containing several pipeline configurations.
        """

        if not pipelines_cfg:
            raise Exception(
                'Received an empty dictionary.' 
                'Please ensure that pipline configuration is properly configured in yaml'
            )
        self.pipelines_cfg = pipelines_cfg
        self.pipelines = {}

    def build_single_pipeline(self, pipeline_cfg: dict) -> Pipeline:
        """
        Build an sklearn pipeline object using provided configuration.
        When a pipeline step in the configuration has a value for "cols", a column transformer is used
        to ensure that the step is applied only for specific columns - the rest are passed through.

        :param pipeline_cfg: Dictionary containing the pipeline configuration.
        :return: A scikit-learn Pipeline object.
        """

        steps = []
        for step in pipeline_cfg:
            class_obj = load_class(step['class'])
            params = step.get('params', {})
            if step.get('cols'):
                col_transformer = ColumnTransformer(
                    transformers=[
                        (step['name'], class_obj(**params), step['cols'])
                    ],
                    remainder='passthrough'
                )
                steps.append((step['name'], col_transformer))
            else:
                steps.append((step['name'], class_obj(**params)))  
                
        return Pipeline(steps)

    def build_pipelines(self) -> Dict[str, Pipeline]:
        """
        Build a dict of sklearn pipeline objects using provided configuration.

        :return: dict of sklearn Pipeline objects.
        """
        for name, pipeline_cfg in self.pipelines_cfg.items():
            self.pipelines[name] = self.build_single_pipeline(pipeline_cfg)
            
        return self.pipelines
