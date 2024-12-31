import pandas as pd
import os
from src.Iris_classification import logger
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.Iris_classification.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=42)
        
        # for sklearn y should be 1 d vector(rows,) not 1d(rows,1) shape pandas dataframe
        # .values will convert pandas dataframe to numpy array and ravel conevrts it to 1d vector
        model.fit(train_x, train_y.values.ravel())

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))