'''Module for creating a SQLite database'''

import sqlite3
import os


class Database:
    '''Database representation class'''

    def __init__(self, db_name:str='mldb', from_scratch:bool=False) -> None:
        '''Start the main database and create its tables'''

        #remove db if specified to
        if from_scratch and os.path.exists(db_name):
            os.remove(db_name)

        self.connection = sqlite3.connect(db_name) #create a db
        self.cursor = self.connection.cursor()
        self.createTableDatasets()
        self.createTableAPIs()
        self.createTableTunedModels()

    def close(self):
        '''Close cursor and connection'''

        self.cursor.close()
        self.connection.close()

    def createTableDatasets(self):
        '''Create table to store datasets data'''

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Datasets (
                id INTEGER PRIMARY KEY,
                train_path TEXT,
                val_path TEXT,
                test_path TEXT,
                source TEXT,
                date DATE,
                language TEXT
            )
        ''')
        self.fieldsDatasets = ['train_path', 'val_path', 'test_path', 
                               'source', 'date', 'language']

    def createTableAPIs(self):
        '''Create table to store APIs data'''
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS APIs (
                id INTEGER PRIMARY KEY,
                uri TEXT
            )
        ''')
        self.fieldsAPIs = ['uri']

    def createTableTunedModels(self):
        '''Create table to store fine-tuned models data'''
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS TunedModels (
                id INTEGER PRIMARY KEY,
                id_datasets INTEGER NOT NULL,
                id_apis INTEGER,
                model_path TEXT,
                learning_rate REAL,
                lora_rank REAL,
                metric_1 REAL,
                deployed INTEGER DEFAULT 0,
                train_curve_path TEXT,
                val_curve_path TEXT,
                FOREIGN KEY (id_datasets) REFERENCES Datasets(id),
                FOREIGN KEY (id_apis) REFERENCES APIs(id)
            )
        ''')
        self.fieldsAPIs = ['id_datasets', 'id_apis', 'model_path', 'learning_rate',
        'lora_rank', 'metric_1', 'deployed', 'train_curve_path', 'val_curve_path']
