import pandas as pd
import numpy as np
import chardet
from typing import Tuple, Optional
import warnings

class CarregadorDados:
    @staticmethod
    def detectar_codificacao(caminho_arquivo: str) -> str:
        """Detecta a codificação (encoding) do arquivo lendo os primeiros bytes."""
        try:
            with open(caminho_arquivo, 'rb') as f:
                dados_brutos = f.read(10000)
                resultado = chardet.detect(dados_brutos)
                codificacao = resultado['encoding'] or 'utf-8'
                return codificacao
        except Exception:
            return 'utf-8'
            
    @staticmethod
    def detectar_separador(caminho_arquivo: str, codificacao: str) -> str:
        """Identifica qual caractere separa as colunas."""
        try:
            with open(caminho_arquivo, 'r', encoding=codificacao) as f:
                primeira_linha = f.readline()
                segunda_linha = f.readline()
                
            linhas = [primeira_linha, segunda_linha] if segunda_linha else [primeira_linha]
            
            pontuacao_separadores = {}
            for sep in [',', ';', '\t', '|']:
                pontuacoes = []
                for linha in linhas:
                    if linha.strip():
                        contagem = linha.count(sep)
                        pontuacoes.append(contagem)
                if pontuacoes:
                    pontuacao_separadores[sep] = min(pontuacoes)
                    
            if pontuacao_separadores:
                melhor_separador = max(pontuacao_separadores, key=pontuacao_separadores.get)
                if pontuacao_separadores[melhor_separador] > 0:
                    return melhor_separador
                    
            return ','
        except Exception:
            return ','
            
    def carregar_dados(self, caminho_arquivo: str) -> Tuple[pd.DataFrame, str, str]:
        """Carrega o arquivo CSV, detectando automaticamente encoding e separador."""
        try:
            codificacao = self.detectar_codificacao(caminho_arquivo)
            separador = self.detectar_separador(caminho_arquivo, codificacao)
            
            df = pd.read_csv(
                caminho_arquivo, 
                sep=separador, 
                encoding=codificacao, 
                low_memory=False,
                on_bad_lines='skip'
            )
            
            if df.empty:
                raise ValueError("O arquivo CSV está vazio")
                
            return df, separador, codificacao
            
        except UnicodeDecodeError:
            try:
                # Tenta fallback para latin-1, caso utf-8 falhe
                df = pd.read_csv(caminho_arquivo, sep=separador, encoding='latin-1', low_memory=False)
                return df, separador, 'latin-1'
            except Exception as e:
                raise ValueError(f"Erro ao ler arquivo com codificação alternativa: {e}")
                
        except Exception as e:
            raise ValueError(f"Erro ao carregar arquivo: {e}")