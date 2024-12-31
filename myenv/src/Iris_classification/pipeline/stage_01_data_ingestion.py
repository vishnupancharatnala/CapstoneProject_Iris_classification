# this is the first pipeline stage

# mention pipeline

from src.Iris_classification.config.configuration import ConfigurationManager
from src.Iris_classification.components.data_ingestion import DataIngestion
from src.Iris_classification import logger


STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        "you need to call this main function in main.py to trigger the pipeline"
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

# class DataIngestionTrainingPipeline , it does not have any constructor, 
# it has a main function which is used to trigger the pipeline
# it reads the configuration from the configuration manager and then
# creates an object of the DataIngestion class and calls the download_file and extract_zip_file functions
# this will download the data and extract it in the data directory

# config = ConfigurationManager(), it reads the configuration from the configuration manager
# data_ingestion_config = config.get_data_ingestion_config(), it reads the data ingestion configuration
# data_ingestion = DataIngestion(config=data_ingestion_config), it creates an object of the DataIngestion class
# data_ingestion.download_file(), it calls the download_file function of the DataIngestion class
# data_ingestion.extract_zip_file(), it calls the extract_zip_file function of the DataIngestion class



if __name__ == '__main__':
    try:
        # logging the start of the stage, this will help in debugging
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main() # calling the main function, trigger the pipeline
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    # the above code will download the data and extract it in the data directory
    # this is the first stage of the pipeline, we will have multiple stages in the pipeline
    
    
# if __name__ == '__main__':, this is the main block of the code
#  obj = DataIngestionTrainingPipeline(), 
# it creates an object of the DataIngestionTrainingPipeline class
# obj.main(), it calls the main function of the DataIngestionTrainingPipeline class