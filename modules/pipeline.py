'''Module for creating to represent project pipeline'''

from modules.data_utils import DataUtils
from modules.database import Database
import datetime
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
import os
import uuid
import os
import subprocess
import sys
from modules.api_templates import APITemplates
import socket
from datasets import load_dataset


class Pipeline:
    '''Pipeline representation class'''

    def __init__(self, db: Database) -> None:
        self.db = db

    def registerDataset(self, input_path: str, source: str, date: datetime.date, language: str, name: str):
        '''Function to implement pipeline for registering a new dataset'''

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

    def _tokenizeSplits(self, tokenizer, dataset):
        '''Tokenize texts to build dataset splits'''

        #define a pad token if there isn't one
        tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token

        #tokenize inputs and labels
        def tokenize_function(examples):
            tokenized_input = tokenizer(examples["text"], padding="max_length", truncation=True)
            tokenized_output = tokenizer(examples["target"], padding="max_length", truncation=True)
            tokenized_examples = {
                "input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
                "labels": tokenized_output["input_ids"]
            }
            return tokenized_examples
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        #split data
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["val"]
        test_dataset = tokenized_datasets["test"]

        return train_dataset, val_dataset, test_dataset
    
    def _train(self, model_name: str, train_dataset, val_dataset, learning_rate: float, ft_option, ranking, artifacts_path):
        '''Train a model given the datasets and configs'''

        #load model
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        #if LoRA, build specific configs
        if ft_option == 'LoRA':
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=ranking,
                lora_alpha=32,
                lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)

        #set parameters for training
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=learning_rate,
            output_dir=artifacts_path,
            num_train_epochs=1,
            logging_steps=2, #log at each batch
            eval_strategy='steps', #evaluate at each step (at each batch, in this case)
            load_best_model_at_end=True
        )

        #train
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset, 
            eval_dataset=val_dataset
        )
        trainer.train()
        
        #save tokenizer and best model
        trainer.save_model()

        return trainer

    def fineTuneModel(self, model_name: str, dataset_id: int, ft_option: str, ranking: int, learning_rate: float):
        '''Function to implement fine-tuning pipeline'''

        #load dataset
        self.db.cursor.execute(f"""
            SELECT train_path, val_path, test_path 
            FROM Datasets
            WHERE id = {dataset_id}
        """)
        paths = self.db.cursor.fetchall()

        #build expected format and load
        data_files = {
            'train': paths[0][0], #train path
            'val': paths[0][1], #val path
            'test': paths[0][2] #test path
        }
        dataset = load_dataset('csv', data_files=data_files)

        #load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        #tokenize
        train_dataset, val_dataset, test_dataset = self._tokenizeSplits(tokenizer, dataset)

        #create folder to save model and its artifacts
        artifacts_path = os.path.join('model_artifacts', str(uuid.uuid4()))
        os.makedirs(artifacts_path, exist_ok=True)

        #fine-tune 
        trainer = self._train(model_name, train_dataset, val_dataset, learning_rate, ft_option, ranking, artifacts_path)

        #to illustrate possibilities of metrics
        test_loss = trainer.evaluate(test_dataset)['eval_loss']

        train_loss = [msr['loss'] for msr in trainer.state.log_history if 'loss' in msr.keys()]
        val_loss = [msr['eval_loss'] for msr in trainer.state.log_history if 'eval_loss' in msr.keys()]
        val_loss = val_loss[:len(train_loss)] #skip last result if val_loss has one entry more than train_loss
        DataUtils().savePickle(os.path.join(artifacts_path, 'train_loss.pkl'), train_loss)
        DataUtils().savePickle(os.path.join(artifacts_path, 'val_loss.pkl'), val_loss)

        #save tokenizer
        tokenizer.save_pretrained(artifacts_path)

        #save metadata
        data = {
            'id_datasets': dataset_id,
            'id_apis': None, #model is not deployed
            'deployed': False, #model is not deployed
            'model_name': model_name,
            'model_path': artifacts_path,
            'learning_rate': learning_rate,
            'lora_rank': ranking,
            'test_loss': test_loss,
            'train_loss_path': os.path.join(artifacts_path, 'train_loss.pkl'),
            'val_loss_path': os.path.join(artifacts_path, 'val_loss.pkl')
        }
        self.db.insertTunedModels(data)

    def deployModel(self, model_id: int):
        '''Function to implement deploy pipeline'''

        #get model path
        self.db.cursor.execute(f"""
            SELECT model_name, model_path FROM TunedModels WHERE id = {model_id}
        """)
        model_name, model_path = self.db.cursor.fetchall()[0]

        #get an available port
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]

        #get code template to deploy
        app_code = APITemplates().getFilledTemplate1(model_name, model_path, port)

        #write app code (API code) on unique .py
        deploy_path = os.path.join('deploys', f'api_{str(uuid.uuid4())}.py')
        with open(deploy_path, 'w') as file:
            file.write(app_code)

        #serve app code (API)
        subprocess.Popen([sys.executable, deploy_path], cwd=os.getcwd())

        #insert API registry on database
        uri = f'http://127.0.0.1:{port}'
        data = {'uri': uri}
        self.db.insertAPIs(data)

        #update status and foreign keys on TunedModels
        self.db.cursor.execute(f"""
            UPDATE TunedModels
                SET deployed = 1, id_apis = {self.db.cursor.lastrowid}
            WHERE id = {model_id}
        """)
        self.db.connection.commit()

