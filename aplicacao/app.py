import streamlit as st
import pandas as pd
import os
import sys
import pickle

from data_loader import CarregadorDados
from feature_extractor import ExtratorMetaCaracteristicas
from cleaner import DataCleaner

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

TRAINING_DIR = os.path.join(ROOT_DIR, 'dados_para_treinamento')
RAW_BANKS_DIR = os.path.join(TRAINING_DIR, 'bancos_brutos')
MASTER_KEY_PATH = os.path.join(TRAINING_DIR, 'gabarito_master.csv')

MODEL_PATH = os.path.join(ROOT_DIR, 'modelos_salvos', 'semantic_type_classifier.pkl')

class SemanticTypeInferenceApp:
    def __init__(self):
        self.data_loader = CarregadorDados()
        self.feature_extractor = ExtratorMetaCaracteristicas()
        self.cleaner = DataCleaner()
        self.model = None
        self.label_encoder = None
        self.feature_columns = []
        self.reference_dictionaries = {} # Armazena os gabaritos carregados
        self.load_model()
        
    def load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                st.error(f"Modelo não encontrado em: {MODEL_PATH}")
                return

            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data.get('feature_columns', [])
        except Exception as e:
            st.error(f"Erro ao carregar modelo IA: {e}")
    
    def load_dictionaries_from_training_data(self, target_columns):
        """
        Lê o 'gabarito_master.csv' para descobrir qual arquivo de treino
        contém os valores válidos para as colunas do usuário.
        """
        if not os.path.exists(MASTER_KEY_PATH):
            return

        # 1. Carrega o mapa (Coluna -> Nome do Arquivo)
        try:
            df_master = pd.read_csv(MASTER_KEY_PATH)
            # Cria dicionário: {'bairro': ['imoveis_aluguel.csv', ...], 'setor': ['vendas.csv']}
            master_map = {}
            for _, row in df_master.iterrows():
                col = str(row['nome_coluna']).strip()
                fname = str(row['nome_arquivo']).strip()
                if col not in master_map: master_map[col] = []
                master_map[col].append(fname)
        except Exception as e:
            print(f"Erro ao ler master key: {e}")
            return

        # 2. Busca gabaritos para as colunas do usuário
        dictionaries = {}
        
        for col in target_columns:
            if col in master_map:
                # Pega o primeiro arquivo que contém essa coluna
                target_file = master_map[col][0]
                full_path = os.path.join(RAW_BANKS_DIR, target_file)
                
                if os.path.exists(full_path):
                    try:
                        # Lê apenas a coluna necessária
                        df_ref = pd.read_csv(full_path, usecols=[col])
                        # Extrai valores únicos, remove nulos e vazios
                        unique_vals = df_ref[col].dropna().astype(str).unique().tolist()
                        unique_vals = [v for v in unique_vals if v.strip()]
                        
                        if unique_vals:
                            dictionaries[col] = unique_vals
                    except Exception as e:
                        print(f"Erro ao extrair gabarito de {target_file}: {e}")
        
        self.reference_dictionaries = dictionaries

    def predict_column_types(self, df):
        predictions = {}
        feature_list = []
        cols = []
        for col in df.columns:
            f = self.feature_extractor.extrair_caracteristicas(df[col], col)
            feature_list.append(f)
            cols.append(col)
        
        features_df = pd.DataFrame(feature_list).fillna(0)
        # Alinha colunas
        for c in self.feature_columns:
            if c not in features_df.columns: features_df[c] = 0
        features_df = features_df[self.feature_columns]
        
        try:
            preds = self.model.predict(features_df)
            types = self.label_encoder.inverse_transform(preds)
            predictions = dict(zip(cols, types))
        except:
            predictions = {c: 'DESCONHECIDO' for c in cols}
        return predictions

    def run(self):
        st.set_page_config(page_title="SLD", layout="wide")
        
        st.title("Sistema de limpeza de dados - PET Estatística")
        
        with st.sidebar:
            st.header("Configurações")
            threshold = st.slider("Sensibilidade da Correção", 60, 100, 85, help="Quanto maior, mais rigoroso.")
        
        uploaded_file = st.file_uploader("Carregue seu CSV", type=['csv'])
        
        if uploaded_file:
            # Lógica de Cache para não reprocessar ao mover o slider
            if 'df_raw' not in st.session_state or st.session_state.get('fname') != uploaded_file.name:
                with st.spinner("Analisando estrutura e buscando gabaritos..."):
                    # Carrega
                    df, _, _ = self.data_loader.carregar_dados(uploaded_file)
                    st.session_state['df_raw'] = df
                    st.session_state['fname'] = uploaded_file.name
                    
                    # Prediz Tipos
                    preds = self.predict_column_types(df)
                    st.session_state['preds'] = preds
                    
                    # Carrega Gabaritos Baseados no Treino
                    self.load_dictionaries_from_training_data(df.columns.tolist())
                    st.session_state['dicts'] = self.reference_dictionaries
            
            df = st.session_state['df_raw']
            preds = st.session_state['preds']
            dicts = st.session_state.get('dicts', {})
            
            # Layout de Revisão
            c1, c2 = st.columns([3, 1])
            with c1: 
                st.dataframe(df.head(), use_container_width=True)
            
            with c2:
                st.subheader("Diagnóstico")
                st.write(f"**Linhas:** {len(df)}")
                st.write(f"**Colunas:** {len(df.columns)}")
                
                found = len(dicts)
                if found > 0:
                    st.success(f" Coluna(s) com gabarito: {found}")
                    with st.expander("Ver detalhes"):
                        st.write("Usando dados de treino para corrigir:", list(dicts.keys()))
                else:
                    st.warning("Nenhum gabarito encontrado. Usando método de frequência.")

            # Botão de Execução
            if st.button("Limpar dados", type="primary"):
                with st.spinner("Limpando os dados..."):
                    df_clean, report = self.cleaner.clean_dataset(
                        df, 
                        preds, 
                        reference_dictionaries=dicts, 
                        similarity_threshold=threshold
                    )
                
                st.success("Limpeza finalizada!")
                
                # Exibe Relatório
                st.subheader("Resumo das correções")
                report_rows = []
                for col, stats in report['columns_cleaned'].items():
                    if stats['rows_corrected'] > 0 or stats['rows_removed'] > 0:
                        report_rows.append({
                            "Coluna": col,
                            "Método": stats['method'],
                            "Corrigidos": stats['rows_corrected'],
                            "Removidos (Nulos)": stats['rows_removed']
                        })
                
                if report_rows:
                    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
                else:
                    st.info("Os dados já parecem limpos ou nenhuma correção atingiu o limiar de confiança.")
                
                # Download
                csv_data = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Baixar CSV limpo",
                    csv_data,
                    "dados_limpos.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    app = SemanticTypeInferenceApp()
    app.run()