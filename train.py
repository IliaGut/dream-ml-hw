import pandas as pd
import joblib
import argparse

from pipeline_builder import PipelineBuilder
from pipeline_selector import PipelineSelector
from utils import load_yaml_config, google_drive_csv_to_df

def run_training(train_cfg_path: str, trained_model_path: str) -> None:
    """
    main training process

    :param train_cfg_path: path to a yaml configuration file
    :param trained_model_path: path to for saving the model
    """

    train_config = load_yaml_config(file_path=train_cfg_path)

    field_types = train_config.get('field_types', {})
    categorical_features = field_types.get('categorical_features', [])
    numerical_features = field_types.get('numerocal_features', []) 
    target = field_types.get('target', None)

    df = google_drive_csv_to_df(**train_config.get('google_drive_loader', {}))
    pipelines = PipelineBuilder(pipelines_cfg=train_config.get('pipeline_options', {})).build_pipelines()
    pipe_selector = PipelineSelector(pipelines=pipelines, **train_config.get('cv_options'))
    pipe_selector.fit(df[categorical_features + numerical_features], df[target])
    joblib.dump(pipe_selector, trained_model_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cfg_path", type=str, help="train config file path")  
    parser.add_argument("--trained_model_path", type=str, help="trained model file path") 
    args = parser.parse_args()

    run_training(
        train_cfg_path=args.train_cfg_path,
        trained_model_path=args.trained_model_path
    )
