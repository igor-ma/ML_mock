'''Module for creating to represent project pipeline'''

from modules.data_utils import DataUtils
from modules.database import Database
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import os
import uuid
import os
import subprocess
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modules.api_templates import APITemplates


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

        #tokenize the 'text' field
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        #split data
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["val"]
        test_dataset = tokenized_datasets["test"]

        return train_dataset, val_dataset, test_dataset
    
    def _train(self, model_name: str, learning_rate: float, train_dataset, val_dataset, ft_option, ranking, artifacts_path):
        '''Train a model given the datasets and configs'''

        #load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
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
      
    def _computeAccuracy(self, trainer, test_dataset):
        '''Compute model accuracy on a test dataset'''
        predictions = trainer.predict(test_dataset)
        predicted_labels = predictions.predictions.argmax(axis=1)
        true_labels = test_dataset['label']
        accuracy = accuracy_score(true_labels, predicted_labels)
        return accuracy

    def fineTuneModel(self, model_name: str, dataset_id: int, ft_option: str, ranking: int, learning_rate: float):
        '''Function to implement fine-tuning pipeline'''

        #load dataset
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

        #load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        #tokenize
        train_dataset, val_dataset, test_dataset = self._tokenizeSplits(tokenizer, dataset)

        #create folder to save model and its artifacts
        artifacts_path = os.path.join('model_artifacts', str(uuid.uuid4()))
        os.makedirs(artifacts_path, exist_ok=True)

        #fine-tune 
        trainer = self._train(model_name, learning_rate, train_dataset, val_dataset, ft_option, ranking, artifacts_path)

        #evaluate final model
        final_metric = self._computeAccuracy(trainer, test_dataset)

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
            'accuracy': final_metric,
            'train_loss_path': os.path.join(artifacts_path, 'train_loss.pkl'),
            'val_loss_path': os.path.join(artifacts_path, 'val_loss.pkl')
        }
        self.db.insertTunedModels(data)

    def deployModel(self, model_id: int):
        '''Function to implement deploy pipeline'''

        self.db.cursor.execute(f"""
            SELECT model_path FROM TunedModels WHERE id = {model_id}
        """)
        model_path = self.db.cursor.fetchall()[0][0]

        app_code = APITemplates().getFilledTemplate1(model_path, model_path)

        deploy_path = os.path.join('deploys', f'api_{str(uuid.uuid4())}.py')

        #write final app code
        with open(deploy_path, 'w') as file:
            file.write(app_code)

        #run final app code
        subprocess.Popen([sys.executable, deploy_path], cwd=os.getcwd())