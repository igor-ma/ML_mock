'''Module for creating to represent project pipeline'''

from modules.data_utils import DataUtils
from modules.database import Database
import streamlit as st
import datetime


class Pipeline:
    '''Pipeline representation class'''

    def __init__(self, db: Database) -> None:
        self.db = db

    def registerDataset(self, input_path: str, source: str, date: datetime.date, language: str, name: str):
        #get data split paths to save on database
        train_path, val_path, test_path = DataUtils().processDataset(input_path)
        
        #prepare data to save on database
        data = {
            'train_path': train_path,
            'val_path': val_path,
            'test_path': test_path,
            'source': source,
            'date': date,
            'language': language,
            'name': name
        }

        #save
        self.db.insertDatasets(data)

    def fineTuneModel(self, model, ds_option, ft_option, ranking, learning_rate):
        pass

    def deployModel(self):
        pass
