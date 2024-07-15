'''Module for creating a data utilities'''

import random
import os
import uuid


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

        lines = self.readLines(input_path)
    
        random.shuffle(lines)
        
        #set sizes
        total_lines = len(lines)
        train_end = int(train_ratio * total_lines)
        val_end = train_end + int(val_ratio * total_lines)
        
        #split data
        train = lines[:train_end]
        val = lines[train_end:val_end]
        test = lines[val_end:]
        
        dirname = os.path.join(output_path, str(uuid.uuid4()))
        os.makedirs(dirname, exist_ok=True)
        train_path = os.path.join(dirname, 'train.txt')
        val_path = os.path.join(dirname, 'val.txt')
        test_path = os.path.join(dirname, 'test.txt')

        #write data
        self.writeLines(train_path, train)
        self.writeLines(val_path, val)
        self.writeLines(test_path, test)

        return train_path, val_path, test_path