'''Module for creating a Graphic User Interface (GUI)'''

import streamlit as st
from datetime import datetime
from modules.database import Database
from modules.data_utils import DataUtils
from modules.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
import traceback


class GUI:
    '''GUI representation class'''

    def __init__(self, db: Database) -> None:
        '''Start the main GUI using Streamlit'''
        self.db = db
        self.pipeline = Pipeline(db)

        st.title('Mini GenPlat - Fine-tuning e Deploy de LLMs')
        tabs = ['Registro de Datasets', 'Fine-tuning', 'Deploy', 'Dashboard']
        choice = st.sidebar.radio('Selecione a aba:', tabs)

        if choice == 'Registro de Datasets':
            self.registro_datasets()
        elif choice == 'Fine-tuning':
            self.fine_tuning()
        elif choice == 'Deploy':
            self.deploy()
        elif choice == 'Dashboard':
            self.dashboard()

    def registro_datasets(self):
        '''Configure tab for recording datasets'''

        #create tab elements
        st.subheader('Registro de Datasets')
        file_path = st.text_input('Path para o dataset', value='example_dataset.csv')
        name = st.text_input('Nome do dataset')
        source = st.text_input('Fonte dataset')
        date = st.date_input('Data de criação', datetime.today())
        language = st.text_input('Idioma')

        if st.button('Salvar'):
            try:
                self.pipeline.registerDataset(file_path, source, date, language, name)
                st.success('Dados salvos com sucesso.')
            except:
                st.error('Erro de inserção.')
                traceback.print_exc()

    def fine_tuning(self):
        '''Configure tab for running fine-tunings'''

        #list available datasets to fine-tune on
        self.db.cursor.execute('SELECT id, name FROM Datasets')
        datasets = self.db.cursor.fetchall()
        datasets_list = [f"{dataset[0]}, {dataset[1]}" for dataset in datasets]

        #list available models
        available_models = ['t5-base', 't5-small']
        
        #create tab elements
        st.subheader('Fine-tuning')
        model = st.selectbox('Modelo', available_models)
        ds_option = st.selectbox('Datasets disponíveis', datasets_list)
        ft_option = st.radio('Opção', ['Clássico', 'LoRA'])
        
        #enable LoRA specific parameters
        if ft_option == 'LoRA':
            ranking = st.select_slider('Ranking', options=[2, 4, 8, 16, 32, 64, 128, 256])
        else:
            ranking = None
        
        learning_rate = st.number_input('Learning Rate', min_value=0.0, max_value=1.0)
        if st.button('Fine-tune'):
            with st.spinner('Fine-tuning em andamento...'):
                dataset_id = int(ds_option.split(',')[0])
                self.pipeline.fineTuneModel(model, dataset_id, ft_option, ranking, learning_rate)
                st.success('Fine-tuning executado com sucesso.')

    def deploy(self):
        '''Configure tab for deploying models'''
        #list available models to deploy
        self.db.cursor.execute('SELECT model_name, id, id_datasets FROM TunedModels')
        models = self.db.cursor.fetchall()
        models_list = [f"{model[0]}, versão/ID {model[1]}, tunado no dataset ID {model[2]}" for model in models]

        st.subheader('Deploy')
        model_option = st.selectbox('Modelos disponíveis para deploy', models_list)
        if st.button('Deploy'):
            os.makedirs('deploys', exist_ok=True)
            with st.spinner(f'Modelo {model_option} com deploy em andamento.'):
                model_id = int(model_option.split('versão/ID')[1].split(',')[0])
                self.pipeline.deployModel(model_id)
                st.success('Deploy executado com sucesso.')

    def dashboard(self):
        '''Configure tab for dashboard'''

        #load tuned models and related data
        self.db.cursor.execute("""
            SELECT * FROM TunedModels
            JOIN APIs ON TunedModels.id_apis = APIs.id
            JOIN Datasets ON TunedModels.id_datasets = Datasets.id
        """)
        deploys = self.db.cursor.fetchall()

        #create selection box for deployed models
        template = "{} versão/ID {}, tunado no dataset {} versão {}, com deploy ID {}"
        prod_models = [template.format(deploy[3], deploy[0], deploy[20], deploy[13], deploy[11]) for deploy in deploys]
        prod_option = st.selectbox('Modelos disponíveis para deploy', prod_models)
        index_selected = prod_models.index(prod_option)

        #process results
        results = []
        for row in deploys:
            result_dict = {
                'TunedModels_id': row[0],
                'TunedModels_model_name': row[3],
                'TunedModels_learning_rate': row[5],
                'TunedModels_lora_rank': row[6],
                'TunedModels_test_loss': row[7],
                'TunedModels_train_loss_path': row[9],
                'TunedModels_val_loss_path': row[10],
                'APIs_uri': row[12],
                'Datasets_id': row[13],
                'Datasets_name': row[20]
            }
            results.append(result_dict)

        selected_row = results[index_selected]
        model_name = selected_row['TunedModels_model_name']
        model_id = selected_row['TunedModels_id']
        dataset_name = selected_row['Datasets_name']
        dataset_id = selected_row['Datasets_id']
        learning_rate = selected_row['TunedModels_learning_rate']
        lora_rank = selected_row['TunedModels_lora_rank']
        test_loss = selected_row['TunedModels_test_loss']
        api_uri = selected_row['APIs_uri']

        #display information
        st.write(f"Modelo: {model_name} (ID: {model_id})")
        st.write(f"Dataset: {dataset_name} (ID: {dataset_id})")
        st.write(f"Learning Rate: {learning_rate}")
        st.write(f"Lora Rank: {lora_rank}")
        st.write(f"Test Loss: {test_loss:.3f}")
        st.write(f"URI da API: {api_uri}")
        st.write(f"Swagger: {api_uri + '/docs'}")

        #load loss curves
        train_loss = DataUtils().loadPickle(selected_row['TunedModels_train_loss_path'])
        val_loss = DataUtils().loadPickle(selected_row['TunedModels_val_loss_path'])

        #plot loss graph
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Curvas de Loss - Treino e Validação')
        plt.legend()
        st.pyplot(plt)