import yaml
import pandas as pd
from importlib import import_module
from typing import Any, Optional

def load_yaml_config(file_path: str) -> dict:
    """
    load the configuration a yaml.

    :param file_path: Path to the configuration file.
    :return: Parsed configuration dictionary.
    """

    with open(file_path, 'r') as f:
        
        return yaml.safe_load(f)

def load_class(cls_string: str) -> Any:
    """
    load a class from a string.

    :param full_class_string: String representing the full class path.
    :return: The class object.
    """

    module_path, class_name = cls_string.rsplit(".", 1)
    module = import_module(module_path)
    
    return getattr(module, class_name)

def google_drive_csv_to_df(link: str, index_col: Optional[str] = None, dropna_cols: Optional[list] = None) -> pd.DataFrame:
    """
    read a csv in google drive into a pandas DataFrame

    :param link: a google drive link
    :param index_col: a column that should be set as index
    """

    try:
        file_id = link.split('/d/')[1].split('/')[0]
    except IndexError:
        raise ValueError('invalid Google Drive link')
    url = f'https://drive.google.com/uc?id={file_id}'
    df = pd.read_csv(url)
    if index_col:
        df.set_index(index_col, inplace=True)
    if dropna_cols:
        df.dropna(subset=dropna_cols, inplace=True)

    return df
