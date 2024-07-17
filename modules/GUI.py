'''Module for creating a Graphic User Interface (GUI)'''

import streamlit as st
from datetime import datetime
from modules.database import Database
from modules.data_utils import DataUtils
from modules.pipeline import Pipeline


class GUI:
    '''GUI representation class'''

    def __init__(self, db: Database) -> None:
        '''Start the main GUI using Streamlit'''
        self.db = db
        self.pipeline = Pipeline(db)

        st.title('Fine-tuning e Deploy de LLMs para Classificação de Sentimentos')
        st.markdown("""
            O dataset (.csv) deve conter os campos 'text' e 'label', e suas labels precisam ser binárias (0 ou 1).
        """)
        tabs = ['Registro de Datasets', 'Fine-tuning', 'Deploy', 'Swagger', 'Dashboard']
        choice = st.sidebar.radio('Selecione a aba:', tabs)

        if choice == 'Registro de Datasets':
            self.registro_datasets()
        elif choice == 'Fine-tuning':
            self.fine_tuning()
        elif choice == 'Deploy':
            self.deploy()
        elif choice == 'Swagger':
            st.subheader('Aba Swagger')
        elif choice == 'Dashboard':
            st.subheader('Aba Dashboard')

    def registro_datasets(self):
        '''Configure tab for recording datasets'''

        #create tab elements
        st.subheader('Registro de Datasets')
        file_path = st.file_uploader('Selecione o arquivo de dados (.csv)')
        name = st.text_input('Nome do dataset')
        source = st.text_input('Fonte do dado')
        date = st.date_input('Data de criação', datetime.today())
        language = st.text_input('Idioma')

        if st.button('Salvar'):
            try:
                self.pipeline.registerDataset(file_path.name, source, date, language, name)
                st.success('Dados salvos com sucesso.')
            except:
                st.error('Erro de inserção.')

    def fine_tuning(self):
        '''Configure tab for running fine-tunings'''

        #list available datasets to fine-tune on
        self.db.cursor.execute('SELECT id, name FROM Datasets')
        datasets = self.db.cursor.fetchall()
        datasets_list = [f"{dataset[0]}, {dataset[1]}" for dataset in datasets]

        #list available models
        available_models = ['bert-base-uncased', 'bert-base-cased', 'distilbert-base-uncased',
                            'roberta-base', 'albert-base-v1']
        
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
        
        learning_rate = st.number_input('Learning Rate', min_value=0.0, max_value=1.0, step=0.01)
        if st.button('Fine-tune'):
            with st.spinner('Fine-tuning em andamento...'):
                self.pipeline.fineTuneModel(model, ds_option, ft_option, ranking, learning_rate)
                st.success('Fine-tuning executado com sucesso.')

    def deploy(self):
        '''Configure tab for deploying models'''

        st.subheader('Deploy')
        models = st.selectbox('Modelos disponíveis para deploy', ['Modelo 1', 'Modelo 2', 'Modelo 3'])
        if st.button('Deploy'):
            st.success(f'Modelo {models} com deploy em andamento.')


