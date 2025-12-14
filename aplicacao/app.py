import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importações atualizadas com os nomes em português
from data_loader import CarregadorDados
from feature_extractor import ExtratorMetaCaracteristicas
from cleaner import LimpadorDados

class AplicacaoInferenciaTipoSemantico:
    def __init__(self):
        self.carregador = CarregadorDados()
        self.extrator = ExtratorMetaCaracteristicas()
        self.limpador = LimpadorDados()
        self.modelo = None
        self.codificador_labels = None
        self.colunas_features = []
        self.carregar_modelo()
        
    def carregar_modelo(self):
        """Carrega o modelo de ML treinado e seus artefatos (encoder, colunas)."""
        try:
            caminho_modelo = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'modelos_salvos',
                'semantic_type_classifier.pkl'
            )
            with open(caminho_modelo, 'rb') as f:
                dados_modelo = pickle.load(f)
            self.modelo = dados_modelo['model']
            self.codificador_labels = dados_modelo['label_encoder']
            self.colunas_features = dados_modelo.get('feature_columns', [])
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            st.stop()

    def salvar_feedback(self, nome_coluna, tipo_predito, tipo_correto, nome_arquivo):
        """Registra correções do usuário em um CSV para re-treinamento futuro."""
        arquivo_feedback = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dados_para_treinamento',
            'feedback_loop.csv'
        )
        arquivo_existe = os.path.isfile(arquivo_feedback)
        with open(arquivo_feedback, 'a', newline='', encoding='utf-8') as f:
            escritor = csv.writer(f)
            if not arquivo_existe:
                escritor.writerow(['data_feedback', 'nome_arquivo', 'nome_coluna', 'tipo_predito', 'tipo_correto'])
            escritor.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), nome_arquivo, nome_coluna, tipo_predito, tipo_correto])
            
    def prever_tipos_colunas(self, df: pd.DataFrame) -> dict:
        """Gera features para o DataFrame e utiliza o modelo para prever tipos semânticos."""
        predicoes = {}
        lista_features = []
        nomes_colunas = []
        
        for coluna in df.columns:
            feats = self.extrator.extrair_caracteristicas(df[coluna], coluna)
            lista_features.append(feats)
            nomes_colunas.append(coluna)
            
        df_features = pd.DataFrame(lista_features)
        df_features = df_features.fillna(0)
        
        # Garante que todas as colunas esperadas pelo modelo existam
        cols_faltantes = set(self.colunas_features) - set(df_features.columns)
        for c in cols_faltantes:
            df_features[c] = 0
        df_features = df_features[self.colunas_features]
        
        try:
            predicoes_codificadas = self.modelo.predict(df_features)
            tipos_preditos = self.codificador_labels.inverse_transform(predicoes_codificadas)
            for nome_col, tipo_pred in zip(nomes_colunas, tipos_preditos):
                predicoes[nome_col] = tipo_pred
        except Exception as e:
            st.error(f"Erro durante a predição: {e}")
            for nome_col in nomes_colunas:
                predicoes[nome_col] = 'DESCONHECIDO'
        return predicoes
        
    def executar(self):
        """Função principal que renderiza a interface Streamlit."""
        st.set_page_config(page_title="Inferência Semântica", layout="wide")
        st.title("Sistema de Inferência e Limpeza de Dados")
        
        with st.sidebar:
            st.header("Configurações de Limpeza")
            limiar_fuzzy = st.slider(
                "Sensibilidade da Correção (Fuzzy)", 
                min_value=50, max_value=100, value=85,
                help="Valores maiores exigem mais similaridade para corrigir erros de digitação."
            )
            st.markdown("---")
            st.header("Ciclo de Feedback")
            st.info("Suas correções manuais serão salvas para re-treinar o modelo futuramente.")

        arquivo_upload = st.file_uploader("Carregue seu arquivo CSV", type=['csv'])
        
        if arquivo_upload is not None:
            # Lógica de cache para não recarregar/reprocessar a cada interação da UI
            if 'df_raw' not in st.session_state or st.session_state.get('nome_arquivo_upload') != arquivo_upload.name:
                try:
                    with st.spinner("Carregando arquivo..."):
                        caminho_temp = f"temp_{arquivo_upload.name}"
                        with open(caminho_temp, "wb") as f:
                            f.write(arquivo_upload.getbuffer())
                            
                        df, separador, codificacao = self.carregador.carregar_dados(caminho_temp)
                        os.remove(caminho_temp)
                        
                        st.session_state['df_raw'] = df
                        st.session_state['nome_arquivo_upload'] = arquivo_upload.name
                        st.session_state['previsoes_tipo'] = self.prever_tipos_colunas(df)
                except Exception as e:
                    st.error(f"Erro: {e}")
                    return
            
            df = st.session_state['df_raw']
            previsoes_tipo = st.session_state['previsoes_tipo']
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Pré-visualização")
                st.dataframe(df.head(), use_container_width=True)
            with col2:
                st.subheader("Tipos Identificados")
                correcoes = {}
                for col in df.columns:
                    pred = previsoes_tipo.get(col, 'DESCONHECIDO')
                    tipos_possiveis = list(self.codificador_labels.classes_) if self.codificador_labels else [pred]
                    
                    novo_tipo = st.selectbox(
                        f"{col}", 
                        options=tipos_possiveis, 
                        index=tipos_possiveis.index(pred) if pred in tipos_possiveis else 0,
                        key=f"sel_{col}"
                    )
                    
                    if novo_tipo != pred:
                        correcoes[col] = (pred, novo_tipo)
                        previsoes_tipo[col] = novo_tipo

                if correcoes:
                    if st.button("Salvar Feedback de Tipos"):
                        for col, (antigo, novo) in correcoes.items():
                            self.salvar_feedback(col, antigo, novo, arquivo_upload.name)
                        st.success("Feedback salvo com sucesso!")

            if st.button("Executar Limpeza", type="primary"):
                with st.spinner("Limpando dados..."):
                    df_limpo, relatorio = self.limpador.limpar_dataset(df, previsoes_tipo, limiar_fuzzy)
                    
                st.subheader("Resultado da Limpeza")
                st.dataframe(df_limpo, use_container_width=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Linhas Corrigidas", relatorio['total_linhas_corrigidas'])
                c2.metric("Linhas Removidas", relatorio['total_linhas_removidas'])
                c3.metric("Datas Formatadas", relatorio['total_datas_formatadas'])
                
                st.download_button(
                    label="Baixar CSV Limpo",
                    data=df_limpo.to_csv(index=False, sep=';'),
                    file_name="dataset_limpo.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    app = AplicacaoInferenciaTipoSemantico()
    app.executar()