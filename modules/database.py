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
                language TEXT,
                name TEXT
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
                model_name TEXT,
                model_path TEXT,
                learning_rate REAL,
                lora_rank REAL,
                accuracy REAL,
                deployed INTEGER DEFAULT 0,
                train_loss_path TEXT,
                val_loss_path TEXT,
                FOREIGN KEY (id_datasets) REFERENCES Datasets(id),
                FOREIGN KEY (id_apis) REFERENCES APIs(id)
            )
        ''')
        self.fieldsTunedModels = ['id_datasets', 'id_apis', 'model_name', 'model_path', 'learning_rate',
        'lora_rank', 'accuracy', 'deployed', 'train_loss_path', 'val_loss_path']
    
    def _imputFields(self, fields: list, data: dict):
        '''Imput fields um data dict that might be missing according to fields list'''

        for field in fields:
            if field not in data:
                data[field] = None
        return data

    def insertDatasets(self, data: dict):
        '''Insert row on table Datasets'''

        data = self._imputFields(self.fieldsDatasets, data)
        self.cursor.execute("""
            INSERT INTO Datasets (train_path, val_path, test_path, source, date, language, name)
            VALUES (:train_path, :val_path, :test_path, :source, :date, :language, :name)
        """, data)
        self.connection.commit()
        if self.cursor.lastrowid is not None:
            print(f"ID da última linha inserida: {self.cursor.lastrowid}")

    def insertAPIs(self, data: dict):
        '''Insert row on table APIs'''
        
        data = self._imputFields(self.fieldsAPIs, data)
        self.cursor.execute("""
            INSERT INTO APIs (uri)
            VALUES (:uri)
        """, data)
        self.connection.commit()
        if self.cursor.lastrowid is not None:
            print(f"ID da última linha inserida: {self.cursor.lastrowid}")

    def insertTunedModels(self, data: dict):
        '''Insert row on table TunedModels'''
        
        data = self._imputFields(self.fieldsTunedModels, data)
        self.cursor.execute("""
            INSERT INTO TunedModels (id_datasets, id_apis, model_name, model_path, learning_rate,
                                     lora_rank, accuracy, deployed, train_loss_path,
                                        val_loss_path)
            VALUES (:id_datasets, :id_apis, :model_name, :model_path, :learning_rate,
                    :lora_rank, :accuracy, :deployed, :train_loss_path,
                            :val_loss_path)
            """, data)
        self.connection.commit()
        if self.cursor.lastrowid is not None:
            print(f"ID da última linha inserida: {self.cursor.lastrowid}")