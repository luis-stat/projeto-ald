import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, List
import warnings

class ExtratorMetaCaracteristicas:
    def __init__(self):
        # Inicializa o vetorizador TF-IDF para analisar nomes das colunas
        self.vetorizador_tfidf = TfidfVectorizer(
            max_features=50,
            lowercase=True,
            analyzer='char_wb',
            ngram_range=(2, 4)
        )
        self.nomes_colunas_ajustados = False
        
    def _pode_converter_para_numerico(self, valor):
        """Verifica se um valor individual pode ser convertido para número."""
        try:
            if pd.isna(valor): return False
            float(str(valor).replace(',', '.'))
            return True
        except: return False
            
    def _pode_converter_para_data(self, valor):
        """Verifica se um valor individual pode ser convertido para data."""
        try:
            if pd.isna(valor): return False
            pd.to_datetime(str(valor), errors='raise', dayfirst=True)
            return True
        except: return False
        
    def ajustar_nomes_colunas(self, nomes_colunas: List[str]):
        """Treina o vetorizador com os nomes das colunas disponíveis."""
        if nomes_colunas:
            try:
                self.vetorizador_tfidf.fit(nomes_colunas)
                self.nomes_colunas_ajustados = True
            except: self.nomes_colunas_ajustados = False
                
    def extrair_caracteristicas(self, serie: pd.Series, nome_coluna: str = "") -> Dict[str, Any]:
        """Extrai métricas estatísticas e estruturais de uma coluna (Série)."""
        caracteristicas = {}
        serie_limpa = serie.dropna()
        total = len(serie)
        nao_nulos = len(serie_limpa)
        
        # Métricas básicas
        caracteristicas['non_null_ratio'] = nao_nulos / total if total > 0 else 0
        caracteristicas['cardinality_ratio'] = serie_limpa.nunique() / nao_nulos if nao_nulos > 0 else 0
        
        # Amostragem para performance
        amostra = serie_limpa.sample(min(len(serie_limpa), 1000), random_state=42) if nao_nulos > 0 else serie_limpa
        
        # Verificação de tipos de conteúdo
        caracteristicas['numeric_ratio'] = amostra.apply(self._pode_converter_para_numerico).sum() / len(amostra) if len(amostra) > 0 else 0
        caracteristicas['date_ratio'] = amostra.apply(self._pode_converter_para_data).sum() / len(amostra) if len(amostra) > 0 else 0
        
        # Comprimento médio das strings
        tam_strings = amostra.astype(str).str.len()
        caracteristicas['avg_len'] = tam_strings.mean() if not tam_strings.empty else 0
        
        # Mapa de tipos de dados pandas
        mapa_dtype = {'object': [1,0,0], 'int64': [0,1,0], 'float64': [0,0,1]}
        feats_d = mapa_dtype.get(str(serie.dtype), [0,0,0])
        caracteristicas['is_obj'], caracteristicas['is_int'], caracteristicas['is_float'] = feats_d
        
        # Embeddings do nome da coluna (se disponível)
        if self.nomes_colunas_ajustados and nome_coluna:
            try:
                emb = self.vetorizador_tfidf.transform([nome_coluna]).toarray()[0]
                for i, v in enumerate(emb): caracteristicas[f'nm_emb_{i}'] = v
            except:
                for i in range(50): caracteristicas[f'nm_emb_{i}'] = 0
        else:
            for i in range(50): caracteristicas[f'nm_emb_{i}'] = 0
                
        return caracteristicas