import pandas as pd
import sys
import os
from rapidfuzz import process, fuzz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aplicacao.cleaner import LimpadorDados

data = {
    "cidade": ["Fortaleza", "bahi a", "São Paula", "Sao aluoxxx", "Sao Paluo", "BAhia", "fortaleza ce"],
    "estado": ["Ce", "ceara", "SaoPualo", "sao paulo", "SP", "sp", "Ceara"],
    "cor": ["Azull", "verMElho", "pret a", "bracno", "verde", "", "Preto "],
    "universidade": [
        "Universidade Federal do Ceara-UFC", 
        "Universidade F Ceará",
        "Univercidade Feral do cceatá", 
        "universidade de Sãoaulo",
        " USP", 
        "univ de sao paulo", 
        "UFC"
    ],
    "renda": [2000, 5000, 4000, 3000, 1000, 1000, 2500],
    "filhos": [2, 3, 1, 0, 4, 2, 1]
}

df_raw = pd.DataFrame(data)

lista_padroes = {
    "cidade": {"padroes": ["Fortaleza", "São Paulo", "Bahia"], "limite": 60},
    "estado": {"padroes": ["Ceará", "São Paulo"], "limite": 80},
    "cor": {"padroes": ["Azul", "Vermelho", "Preto", "Branco"], "limite": 80},
    "universidade": {
        "padroes": ["Universidade Federal do Ceará - UFC", "Universidade de São Paulo - USP"], 
        "limite": 55
    }
}

# Nova função
def corrigir_dataframe_dicionario(df, regras):
    """
    Corrige o dataframe comparando valores contra uma lista fixa de padrões (Golden Source).
    """
    df_out = df.copy()
    
    for col, config in regras.items():
        if col not in df_out.columns:
            continue
            
        padroes = config['padroes']
        limite = config['limite']
        
        # Função interna para aplicar em cada linha
        def match_value(val):
            if pd.isna(val) or str(val).strip() == "":
                return val
            
            # process.extractOne retorna (melhor_match, score, index)
            match, score, _ = process.extractOne(
                str(val), 
                padroes, 
                scorer=fuzz.token_sort_ratio
            )
            
            if score >= limite:
                return match
            else:
                return val
                
        df_out[col] = df_out[col].apply(match_value)
        
    return df_out

print("=== Banco sintético ===")
print(df_raw[["cidade", "estado", "cor", "universidade"]])

cleaner = LimpadorDados()
df_cleaner = df_raw.copy()

for col in ["cidade", "estado", "cor", "universidade"]:
    df_cleaner[col] = df_cleaner[col].apply(cleaner.para_titulo_br)
    # Aplica correção fuzzy automática (self-correction)
    df_cleaner[col], _ = cleaner.correcao_fuzzy(df_cleaner[col])

print("\n=== Método inicial ===")
print(df_cleaner[["cidade", "estado", "cor", "universidade"]])

df_dict = corrigir_dataframe_dicionario(df_raw, lista_padroes)

print("\n=== Método novo ===")
print(df_dict[["cidade", "estado", "cor", "universidade"]])