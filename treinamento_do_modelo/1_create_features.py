import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aplicacao.feature_extractor import MetaFeatureExtractor

class FeatureCreator:
    def __init__(self):
        self.feature_extractor = MetaFeatureExtractor()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def load_gabarito(self) -> pd.DataFrame:
        gabarito_path = os.path.join(self.base_dir, 'dados_para_treinamento', 'gabarito_master.csv')
        if not os.path.exists(gabarito_path):
            raise FileNotFoundError(f"Arquivo gabarito não encontrado: {gabarito_path}")
        return pd.read_csv(gabarito_path)
        
    def load_training_files(self) -> List[str]:
        banco_bruto_path = os.path.join(self.base_dir, 'dados_para_treinamento', 'bancos_brutos')
        if not os.path.exists(banco_bruto_path):
            raise FileNotFoundError(f"Diretório banco_bruto não encontrado: {banco_bruto_path}")
            
        csv_files = [f for f in os.listdir(banco_bruto_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("Nenhum arquivo CSV encontrado no diretório banco_bruto")
            
        return [os.path.join(banco_bruto_path, f) for f in csv_files]
        
    def detect_separator(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                
            if first_line.count(';') > first_line.count(','):
                return ';'
            else:
                return ','
        except Exception:
            return ','
            
    def create_features_dataset(self) -> pd.DataFrame:
        gabarito_df = self.load_gabarito()
        training_files = self.load_training_files()
        
        all_features = []
        all_column_names = []
        
        for file_path in training_files:
            filename = os.path.basename(file_path)
            try:
                separator = self.detect_separator(file_path)
                df = pd.read_csv(file_path, sep=separator, encoding='utf-8', low_memory=False)
                
                for column in df.columns:
                    all_column_names.append(column)
                    
            except Exception as e:
                warnings.warn(f"Erro ao ler arquivo {filename}: {e}")
                continue
                
        self.feature_extractor.fit_column_names(all_column_names)
        
        for file_path in training_files:
            filename = os.path.basename(file_path)
            try:
                separator = self.detect_separator(file_path)
                df = pd.read_csv(file_path, sep=separator, encoding='utf-8', low_memory=False)
                
                for column in df.columns:
                    gabarito_row = gabarito_df[
                        (gabarito_df['nome_arquivo'] == filename) & 
                        (gabarito_df['nome_coluna'] == column)
                    ]
                    
                    if not gabarito_row.empty:
                        tipo_real = gabarito_row['tipo_real'].iloc[0]
                        
                        features = self.feature_extractor.extract_features(df[column], column)
                        features['target'] = tipo_real
                        features['nome_arquivo'] = filename
                        features['nome_coluna'] = column
                        
                        all_features.append(features)
                        
            except Exception as e:
                warnings.warn(f"Erro ao processar arquivo {filename}: {e}")
                continue
                
        if not all_features:
            raise ValueError("Nenhuma feature foi extraída. Verifique o gabarito e os arquivos de treino.")
            
        features_df = pd.DataFrame(all_features)
        return features_df
        
    def save_features(self, features_df: pd.DataFrame):
        output_path = os.path.join(self.base_dir, 'dados_para_treinamento', 'training_features.csv')
        features_df.to_csv(output_path, index=False)
        print(f"Dataset de features salvo em: {output_path}")
        print(f"Shape do dataset: {features_df.shape}")
        print(f"Distribuição de classes:\n{features_df['target'].value_counts()}")

if __name__ == "__main__":
    try:
        creator = FeatureCreator()
        features_df = creator.create_features_dataset()
        creator.save_features(features_df)
    except Exception as e:
        print(f"Erro durante a criação de features: {e}")
        sys.exit(1)