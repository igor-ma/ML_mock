'''Module for creating a Graphic User Interface (GUI)'''

import streamlit as st
from datetime import datetime
from modules.database import Database
from modules.data_utils import DataUtils


class GUI:
    '''GUI representation class'''

    def __init__(self, db: Database) -> None:
        '''Start the main GUI using Streamlit'''
        self.db = db

        st.title('Case - Fine-tuning e Deploy de LLMs')
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

        st.subheader('Registro de Datasets')
        txt_path = st.file_uploader('Selecione o arquivo de dados (.txt)')
        source = st.text_input('Fonte do dado')
        date = st.date_input('Data de criação', datetime.today())
        language = st.text_input('Idioma')

        if st.button('Salvar'):
            train_path, val_path, test_path = DataUtils().processDataset(
                txt_path.name)
            
            data = {
                'train_path': train_path,
                'val_path': val_path,
                'test_path': test_path,
                'source': source,
                'date': date,
                'language': language
            }
            self.db.insertDatasets(data)
            
            st.success('Dados salvos com sucesso.')

    def fine_tuning(self):
        '''Configure tab for running fine-tunings'''

        st.subheader('Fine-tuning')
        model = st.text_input('Modelo')
        datasets = st.selectbox('Datasets disponíveis', ['Dataset 1', 'Dataset 2', 'Dataset 3'])
        option = st.radio('Opção', ['Clássico', 'LoRA'])
        
        #enable LoRA specific parameters
        if option == 'LoRA':
            ranking = st.select_slider('Ranking', options=[2, 4, 8, 16, 32, 64, 128, 256])
        elif option == 'Clássico':
            pass
        
        learning_rate = st.number_input('Learning Rate', min_value=0.0, max_value=1.0, step=0.01)
        if st.button('Fine-tune'):
            st.success('Fine-tuning em andamento.')

    def deploy(self):
        '''Configure tab for deploying models'''

        st.subheader('Deploy')
        models = st.selectbox('Modelos disponíveis para deploy', ['Modelo 1', 'Modelo 2', 'Modelo 3'])
        if st.button('Deploy'):
            st.success(f'Modelo {models} com deploy em andamento.')


