# metalsitenn/utils.py
'''
* Author: Evan Komp
* Created: 11/26/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pandas as pd
from dataclasses import dataclass
import re
from typing import List, Tuple, Iterator

@dataclass
class ParamsObj:
    """Wraps dict of dicts to allow attribute access to nested dicts."""
    def __init__(self, upper_dict):
        for k, v in upper_dict.items():
            if isinstance(v, dict):
                setattr(self, k, ParamsObj(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __repr__(self):
        return str(self.__dict__)
    
    @property
    def param_names(self):
        return list(self.__dict__.keys())
    
def get_emission_time_job_from_codecarbon_log(emissions_file: str, project_name: str) -> pd.DataFrame:
    """
    Parse the CodeCarbon emissions log file to extract the time and job ID for each emission. Assumes most recent emissions are at the end of the file.
    
    Args:
        emissions_file (str): Path to the CodeCarbon emissions log file.
        project_name (str): Name of the project to extract emissions for.
    
    Returns:
        pd.DataFrame: DataFrame containing the time and job ID for each emission.
    """
    df = pd.read_csv(emissions_file)
    df = df[df["project_name"] == project_name]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by='timestamp', ascending=False)

    if len(df) == 0:
        raise ValueError(f"No emissions found for project {project_name} in file {emissions_file}")

    duration = str(df['duration'].iloc[0])
    emissions = str(df['emissions'].iloc[0])

    return duration, emissions