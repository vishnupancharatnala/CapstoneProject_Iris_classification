from src.Iris_classification import logger

# logger.info("welcome to my capstone stone project on IRIS classification")

from src.Iris_classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Iris_classification.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.Iris_classification.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.Iris_classification.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.Iris_classification.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
# from src.mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

# import configuration manager
from src.Iris_classification.config.configuration import ConfigurationManager

# import data ingestion
from src.Iris_classification.components.data_ingestion import DataIngestion
from src.Iris_classification.components.data_validation import DataValiadtion


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


STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataValidationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e  # check the output in STATUS.txt if true then move to next stage


STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataTransformationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelTrainerTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
