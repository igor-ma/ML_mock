'''Module for creating a data utilities'''

import random
import os
import uuid
import pandas as pd
import pickle as pkl

random.seed(123)


class DataUtils:
    '''Data utilities representation class'''

    def __init__(self) -> None:
        pass

    def readLines(self, input_path: str, encoding: str='utf-8'):
        '''Read a .txt lines and return them'''
        with open(input_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        return lines
    
    def writeLines(self, output_path: str, lines: list, encoding: str='utf-8'):
        '''Write lines in a .txt'''
        with open(output_path, 'w', encoding=encoding) as f:
            f.writelines(lines)
    
    def processDataset(self, input_path: str, output_path: str='data', 
                       train_ratio: float=0.6, val_ratio:float=0.2):
        '''Process a .txt dataset'''

        #read dataset
        df = pd.read_csv(input_path)

        #shuffle
        df = df.sample(frac=1).reset_index(drop=True)
    
        #set sizes
        total_size = df.shape[0]
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        #split data
        df_train = df.iloc[:train_size]
        df_val = df.iloc[train_size:train_size + val_size]
        df_test = df.iloc[train_size + val_size:]
        
        #save
        dirname = os.path.join(output_path, str(uuid.uuid4()))
        os.makedirs(dirname, exist_ok=True)
        train_path = os.path.join(dirname, 'train.csv')
        val_path = os.path.join(dirname, 'val.csv')
        test_path = os.path.join(dirname, 'test.csv')

        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)

        return train_path, val_path, test_path
    
    def savePickle(self, output_path: str, obj):
        '''Save a Python object using pickle'''
        with open(output_path, 'wb') as f:
            pkl.dump(obj, f)