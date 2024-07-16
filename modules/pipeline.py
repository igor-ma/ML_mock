'''Module for creating to represent project pipeline'''

from modules.data_utils import DataUtils
from modules.database import Database
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset


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


    def _train(self, model_name: str, learning_rate: float, train_dataset, val_dataset):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        #set training parameters
        training_args = Seq2SeqTrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            logging_steps=10,
            save_steps=10,
            eval_steps=10,
            output_dir="./results",
            logging_dir="./logs",
            overwrite_output_dir=True,
            save_total_limit=3,
            num_train_epochs=1
        )

        #set trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )

        #fine-tune
        trainer.train()

        #save final model
        trainer.save_model()

        return trainer
      
    def _eval(self, trainer, test_dataset):
        return trainer.evaluate(test_dataset=test_dataset)

    def fineTuneModel(self, model_name: str, ds_option: str, ft_option: str, ranking: int, learning_rate: float):
        train_dataset = load_dataset('text', data_files="data/2c1d6990-c264-4442-bf9c-3776f7d3bb4a/train.txt")
        val_dataset = load_dataset('text', data_files="data/2c1d6990-c264-4442-bf9c-3776f7d3bb4a/val.txt")
        test_dataset = load_dataset('text', data_files="data/2c1d6990-c264-4442-bf9c-3776f7d3bb4a/test.txt")

        #fine-tune 
        trainer = self._train(model_name, learning_rate, train_dataset, val_dataset)

        #evaluate final model
        results = self._eval(trainer, test_dataset)

    def deployModel(self):
        pass
