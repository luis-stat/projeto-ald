# Todos os métodos estão sendo simulados e testados, não são a versão completa

import pandas as pd
import numpy as np
import time
import re
import unicodedata
from rapidfuzz import process, fuzz, utils

np.random.seed(42)
TAMANHO_DATASET = 500000

class BenchmarkMethods:
    def __init__(self, gabarito):
        self.gabarito = gabarito
        # Preparação para o método híbrido
        self.candidates_data = []
        for cand in gabarito:
            norm = self.normalize_text(cand)
            self.candidates_data.append({
                'orig': cand,
                'norm': norm,
                'tokens': set(norm.split())
            })

    def normalize_text(self, text):
        text = str(text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        return re.sub(r'\s+', ' ', text)

    # Método por frequência
    def run_frequency_method(self, data_series):
        # Identifica frequentes (> 1%)
        counts = data_series.value_counts()
        threshold = len(data_series) * 0.01
        frequentes = counts[counts > threshold].index.tolist()
        frequentes_str = [str(x) for x in frequentes] # O Rapidfuzz precisa de string

        def correct(val):
            val_str = str(val)
            if val_str in frequentes_str:
                return val_str
            # Busca match no grupo de frequentes
            match, score, _ = process.extractOne(val_str, frequentes_str, scorer=fuzz.ratio)
            if score > 80:
                return match
            return val_str

        return data_series.apply(correct)

    # Método por dicionários
    def run_dictionary_method(self, data_series):
        def correct(val):
            val_str = str(val)
            # Scorer simples (Ratio/Levenshtein - sei que é a escala, mas não sei se é a padrão)
            match, score, _ = process.extractOne(val_str, self.gabarito, scorer=fuzz.ratio)
            if score > 80:
                return match
            return val_str
        return data_series.apply(correct)

    # Método híbrido ou por hierarquia
    def run_hybrid_method(self, data_series):
        def correct(val):
            query = str(val)
            q_norm = self.normalize_text(query)
            q_tokens = set(q_norm.split())
            
            # Filtro de Token Overlap
            best_overlap = 0
            candidates = []
            
            if not q_tokens:
                return query

            # Verifica overlap
            for cand in self.candidates_data:
                intersection = len(q_tokens.intersection(cand['tokens']))
                if intersection > best_overlap:
                    best_overlap = intersection
                    candidates = [cand]
                elif intersection == best_overlap and intersection > 0:
                    candidates.append(cand)
            
            # Se não achou overlap, usa todos (ou poderia retornar None para ser estrito)
            # Para testar, usei todos caso não tenha overlap para competir com o dicionário
            final_cands = candidates if candidates else self.candidates_data
            
            choices = [c['norm'] for c in final_cands]
            
            # Match final
            match, score, idx = process.extractOne(q_norm, choices, scorer=fuzz.token_sort_ratio)
            
            if score >= 60: # Threshold
                return final_cands[idx]['orig']
            return query

        return data_series.apply(correct)

# Simulação dos dados
gabarito = ["Universidade de São Paulo", "Rio de Janeiro", "Belo Horizonte", 
            "Fortaleza", "Porto Alegre", "Curitiba", "Salvador"]

df = pd.DataFrame({
    'correto': np.random.choice(gabarito, TAMANHO_DATASET),
    'ruido': np.random.uniform(0, 1, TAMANHO_DATASET)
})

def sujar_dados(row):
    correto = row['correto']
    r = row['ruido']
    
    if correto == "Universidade de São Paulo" and r < 0.7:
        return "Univ Sao Paulo"
    if correto == "Rio de Janeiro" and r < 0.6:
        return "rio de janeiro"
    if correto == "Belo Horizonte" and r < 0.2:
        return "BH"
    if r > 0.95:
        return correto[:5] + "xx" # Typo raro
    return correto

df['sujo'] = df.apply(sujar_dados, axis=1)

bench = BenchmarkMethods(gabarito)

print(f"Testando com (N={TAMANHO_DATASET}) dados em Python\n")

# Teste Frequência
t0 = time.perf_counter()
res_freq = bench.run_frequency_method(df['sujo'])
t_freq = time.perf_counter() - t0

# Teste Dicionário
t0 = time.perf_counter()
res_dict = bench.run_dictionary_method(df['sujo'])
t_dict = time.perf_counter() - t0

# Teste Híbrido
t0 = time.perf_counter()
res_hib = bench.run_hybrid_method(df['sujo'])
t_hib = time.perf_counter() - t0

acc_freq = (res_freq == df['correto']).mean() * 100
acc_dict = (res_dict == df['correto']).mean() * 100
acc_hib = (res_hib == df['correto']).mean() * 100

print("Desempenho (Tempo em segundos):")
print(f"1. Método Frequência:  {t_freq:.4f}s")
print(f"2. Método Dicionário:  {t_dict:.4f}s")
print(f"3. Método Híbrido:     {t_hib:.4f}s")

print("\nPrecisão (Acurácia):")
print(f"1. Método Frequência:  {acc_freq:.2f}%")
print(f"2. Método Dicionário:  {acc_dict:.2f}%")
print(f"3. Método Híbrido:     {acc_hib:.2f}%")

# Exemplo BH
print("\nTeste em caso extremo:")
try:
    idx_bh = df[df['sujo'] == 'BH'].index[0]
    print(f"Entrada: 'BH' | Correto: 'Belo Horizonte'")
    print(f"-> Frequência disse: '{res_freq[idx_bh]}'")
    print(f"-> Dicionário disse: '{res_dict[idx_bh]}'")
    print(f"-> Híbrido disse:    '{res_hib[idx_bh]}'")
except IndexError:
    print("Exemplo 'BH' não gerado no random seed atual.")