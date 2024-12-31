from dataclasses import dataclass
from pathlib import Path

# What does config_entity do ? 
# It contains the configuration classes for the different stages 
# of the ML pipeline.
# The configuration classes are dataclasses that store the
# configuration parameters for each stage of the pipeline.
# The configuration classes are used to pass the conf iguration
# parameters to the components of the pipeline.
# The configuration classes are immutable, meaning that once
# the configuration parameters are set, they cannot be changed.


# data ingestion entity : modular coding piece
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

# Class DataIngestionConfig is a dataclass that contains 
# the configuration for the data ingestion process.
# The dataclass decorator is used to create a dataclass
# with the frozen attribute set to True. This means that, 
# the dataclass is immutable.

# The DataIngestionConfig class has four attributes:
# root_dir: Path, source_URL: str, local_data_file: Path, unzip_dir: Path
# root_dir is the root directory where the data is stored.
# source_URL is the URL from where the data is downloaded.
# local_data_file is WHERE the data is stored locally.
# unzip_dir is the directory where the data is unzipped.


# data validation entity : modular coding piece
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

# all_schema is a dictionary containing the schema of the data.
# The DataValidationConfig class has four attributes:
# root_dir: Path, STATUS_FILE: str, unzip_data_dir: Path, all_schema: dict
# root_dir is the root directory where the data is stored.
# STATUS_FILE is the file where the validation status is stored.
# unzip_data_dir is the directory where the data is unzipped.


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

# DataTransformationConfig is a dataclass that contains the configuration for the data transformation process.
# The DataTransformationConfig class has two attributes:
# root_dir: Path, data_path: Path
# root_dir is the root directory where the data is stored.
# data_path is the path to the data file, which is used for data transformation.

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    target_column: str

# ModelTrainerConfig is a dataclass that contains the configuration for the model training process.
# The ModelTrainerConfig class has five attributes:
# root_dir: Path, train_data_path: Path, test_data_path: Path, model_name: str, alpha: float, l1_ratio: float, target_column: str
# root_dir is the root directory where the data is stored.
# train_data_path is the path to the training data file.
# test_data_path is the path to the test data file.
# model_name is the name of the model to be trained.
# alpha is the regularization parameter for the model.
# l1_ratio is the L1 ratio for the model.
# target_column is the target column for the model.

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str

# ModelEvaluationConfig is a dataclass that contains the configuration for the model evaluation process.
# Model Evaluation means evaluating the model on the test data, 
# calculating the metrics, and saving the results.
# The ModelEvaluationConfig class has six attributes:
# root_dir: Path, test_data_path: Path, model_path: Path, all_params: dict, metric_file_name: Path, target_column: str
# root_dir is the root directory where the data is stored.
# test_data_path is the path to the test data file.
# model_path is the path to the trained model file.
# all_params is a dictionary containing all the parameters of the model.
# metric_file_name is the name of the file where the metrics will be saved.
# target_column is the target column for the model.