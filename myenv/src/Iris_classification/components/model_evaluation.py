import os
import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.Iris_classification.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from src.Iris_classification.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        # rmse = np.sqrt(mean_squared_error(actual, pred))
        # mae = mean_absolute_error(actual, pred)
        # r2 = r2_score(actual, pred)
        # return rmse, mae, r2
        accuracy = accuracy_score(actual, pred)
    
        # Precision
        precision = precision_score(actual, pred,average='macro')
        
        # Recall
        recall = recall_score(actual, pred,average="macro")
        
        # F1-Score
        f1 = f1_score(actual, pred,average='macro')
        
        # Confusion Matrix
        cm = confusion_matrix(actual, pred)
        
        # ROC AUC Score (for binary classification, can be used for multi-class with adjustments)
        try:
            roc_auc = roc_auc_score(actual, pred,average='macro', multi_class='ovr')
        except ValueError:
            roc_auc = None  # If ROC AUC can't be computed (e.g., multi-class without adjustments)
        
        return accuracy, precision, recall, f1, cm, roc_auc


    
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)
        

        (accuracy,precision,recall,f1,cm,roc_auc) = self.eval_metrics(test_y, predicted_qualities)
        
        # save_json strictly takes dictionary ,in dictionary values can be list ,str...but not np.array
        # so convert if they are numpy array to list
        
        if isinstance(accuracy, np.ndarray):
            accuracy = accuracy.tolist()
        if isinstance(precision, np.ndarray):
            precision_ = precision.tolist()
        if isinstance(recall, np.ndarray):
            recall = recall.tolist()
        if isinstance(f1, np.ndarray):
            f1= f1.tolist()
        if isinstance(cm, np.ndarray):
            cm = cm.tolist()
        if isinstance(roc_auc, np.ndarray):
            roc_auc = roc_auc.tolist()
        
        # Saving metrics as local
        scores = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,"cm":cm,"roc_auc":roc_auc}
        save_json(path=Path(self.config.metric_file_name), data=scores)