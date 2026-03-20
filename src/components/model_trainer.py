import os 
import sys 

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evalute_model
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression":LinearRegression(),
                "KNN":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Randaom Forest":RandomForestRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(verbose = False)
            }

            model_report:dict = evalute_model(X_train=X_train,y_train=y_train,
                                         X_test=X_test,y_test=y_test,
                                         models=models)
            
            # to get the best model score from dict
            best_model_score = max(model_report.values())

            # to get best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best Model forun on both training and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            final_r2_score = r2_score(y_test,predicted)

            return final_r2_score
        except Exception as e:
            raise CustomException(e,sys)