'''Module for creating to represent project pipeline'''

from modules.data_utils import DataUtils
from modules.database import Database
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


class Pipeline:
    '''Pipeline representation class'''

    def __init__(self, db: Database) -> None:
        self.db = db

    def registerDataset(self, input_path: str, source: str, date: datetime.date, language: str, name: str):
        '''Register a new dataset'''

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

    def _tokenizeSplits(self, model_name: str, dataset):
        '''Tokenize texts to build dataset splits'''

        #load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        #define a pad token if there isn't one
        tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token

        #tokenize the 'text' field
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        #split data
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["val"]
        test_dataset = tokenized_datasets["test"]

        return train_dataset, val_dataset, test_dataset
    
    def _train(self, model_name: str, learning_rate: float, train_dataset, val_dataset):
        '''Train a model given the datasets and configs'''

        #load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        #set parameters for training
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=learning_rate,
            output_dir="./results",
            num_train_epochs=1
        )

        #train
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset, 
            eval_dataset=val_dataset
        )
        trainer.train()
        
        #save final model
        trainer.save_model()

        return trainer
      
    def _eval(self, trainer, test_dataset):
        '''Evaluate a model (inside object trainer) on a test dataset'''
        return trainer.evaluate(eval_dataset=test_dataset)

    def fineTuneModel(self, model_name: str, ds_option: str, ft_option: str, ranking: int, learning_rate: float):
        #load dataset
        dataset_id = ds_option.split(',')[0]
        self.db.cursor.execute(f"""
            SELECT train_path, val_path, test_path 
            FROM Datasets
            WHERE id = {dataset_id}
        """)
        paths = self.db.cursor.fetchall()

        data_files = {
            'train': paths[0][0],
            'val': paths[0][1],
            'test': paths[0][2]
        }
        dataset = load_dataset('csv', data_files=data_files)

        train_dataset, val_dataset, test_dataset = self._tokenizeSplits(model_name, dataset)

        #fine-tune 
        trainer = self._train(model_name, learning_rate, train_dataset, val_dataset)

        #evaluate final model
        results = self._eval(trainer, test_dataset)
        print(results)

    def deployModel(self):
        pass
