import os
from src.Iris_classification import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.Iris_classification.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up
    
    def preprocessing(self):
        data = pd.read_csv(self.config.data_path)
        

        le = LabelEncoder()
        scaler = StandardScaler()
        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']
        for col in columns[:-1]:
            data[[col]] = scaler.fit_transform(data[[col]])

        
        data['species'] = le.fit_transform(data['species'])
        return data
        


    def train_test_spliting(self):
        data = self.preprocessing()

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        # print(train.shape)
        # print(test.shape)