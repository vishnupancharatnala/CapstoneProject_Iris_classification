# first component is data ingestion

import os
import urllib.request as request
import zipfile
from src.Iris_classification import logger
from src.Iris_classification.utils.common import get_size
from src.Iris_classification.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

# Class DataIngestion is a class that is used to download the data and extract it in the data directory.
# It has a constructor that takes the DataIngestionConfig object as an argument.
# self.config = config, it assigns the DataIngestionConfig object to the self.config attribute.

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

# def download_file(self):,
# this function is used to download the data file from the source URL.
# It checks if the file already exists, if it does not exist, it downloads the file using the urllib.request.urlretrieve function.
# It logs the download information using the logger.info function.


    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            
# def extract_zip_file(self):,
# this function is used to extract the zip file into the data directory.
# It creates the unzip directory if it does not exist.
#  It extracts the zip file using the zipfile.ZipFile.extractall function.
# It logs the extraction information using the logger.info function.
# The DataIngestion class is a component of the ML pipeline that is used to download the data and extract it in the data directory.