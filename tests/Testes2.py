import pandas as pd
import numpy as np
import time
import random
import unicodedata
import re
from rapidfuzz import process, fuzz
from typing import List, Dict, Callable
from itertools import product
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. GERADOR DE CAOS (Error Injector)
# ==============================================================================
class ErrorInjector:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def normalize(self, text):
        return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').lower()

    def inject_typo(self, text):
        if len(text) < 2: return text
        idx = random.randint(0, len(text) - 1)
        char = chr(random.randint(97, 122)) # a-z
        return text[:idx] + char + text[idx+1:]

    def inject_token_loss(self, text):
        tokens = text.split()
        if len(tokens) > 1:
            tokens.pop(random.randint(0, len(tokens)-1))
            return " ".join(tokens)
        return text

    def inject_case_noise(self, text):
        return text.lower() if random.random() > 0.5 else text.upper()

    def generate_dataset(self, gabarito: List[str], size: int, noise_level: float, error_type: str) -> pd.DataFrame:
        data = pd.DataFrame({
            'Gabarito': np.random.choice(gabarito, size),
            'is_dirty': False
        })
        
        n_dirty = int(size * noise_level)
        dirty_indices = np.random.choice(data.index, n_dirty, replace=False)
        data.loc[dirty_indices, 'is_dirty'] = True
        
        def apply_noise(row):
            if not row['is_dirty']: return row['Gabarito']
            val = row['Gabarito']
            if error_type == 'typo': return self.inject_typo(val)
            elif error_type == 'token': return self.inject_token_loss(val)
            elif error_type == 'case': return self.inject_case_noise(val)
            elif error_type == 'mixed':
                r = random.random()
                if r < 0.33: return self.inject_typo(val)
                elif r < 0.66: return self.inject_token_loss(val)
                else: return self.inject_case_noise(val)
            return val

        data['Base_Suja'] = data.apply(apply_noise, axis=1)
        return data

# ==============================================================================
# 2. MÉTODOS DE LIMPEZA
# ==============================================================================
class CleaningMethods:
    def __init__(self, gabarito: List[str]):
        self.gabarito = gabarito
        self.gabarito_norm = {self._norm(x): x for x in gabarito}
        self.gabarito_tokens = {self._norm(x): set(self._norm(x).split()) for x in gabarito}

    def _norm(self, text):
        return str(text).lower().strip()

    def metodo_frequencia(self, series: pd.Series) -> pd.Series:
        counts = series.value_counts()
        threshold = len(series) * 0.01
        validos = counts[counts > threshold].index.tolist()
        validos_str = [str(x) for x in validos]
        
        def correct(val):
            val_str = str(val)
            if val_str in validos_str: return val_str
            match, score, _ = process.extractOne(val_str, validos_str, scorer=fuzz.ratio)
            return match if score > 80 else val_str
        return series.apply(correct)

    def metodo_dicionario(self, series: pd.Series) -> pd.Series:
        def correct(val):
            match, score, _ = process.extractOne(str(val), self.gabarito, scorer=fuzz.ratio)
            return match if score > 80 else val
        return series.apply(correct)

    def metodo_hibrido(self, series: pd.Series) -> pd.Series:
        def correct(val):
            val_norm = self._norm(val)
            val_tokens = set(val_norm.split())
            
            candidates = []
            best_overlap = 0
            
            for truth_norm, tokens in self.gabarito_tokens.items():
                overlap = len(val_tokens.intersection(tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    candidates = [truth_norm]
                elif overlap == best_overlap and overlap > 0:
                    candidates.append(truth_norm)
            
            final_pool = [self.gabarito_norm[c] for c in candidates] if candidates else self.gabarito
            
            match, score, _ = process.extractOne(str(val), final_pool, scorer=fuzz.token_sort_ratio)
            return match if score > 60 else val
            
        return series.apply(correct)

# ==============================================================================
# 3. ENGINE ESTATÍSTICA
# ==============================================================================
def run_benchmark():
    gabarito = [
        "Universidade de São Paulo", "Universidade Federal do Rio de Janeiro",
        "Universidade Estadual de Campinas", "Universidade Federal de Minas Gerais",
        "Universidade Federal do Ceará", "Universidade de Brasília",
        "Pontifícia Universidade Católica"
    ]
    
    injector = ErrorInjector()
    cleaner = CleaningMethods(gabarito)
    
    # Grid menor para ser rápido no exemplo
    scenarios = list(product(
        [1000, 5000],
        [0.1, 0.8],
        ['typo', 'token', 'mixed']
    ))
    
    methods = {
        'Frequencia': cleaner.metodo_frequencia,
        'Dicionario': cleaner.metodo_dicionario,
        'Hibrido': cleaner.metodo_hibrido
    }
    
    results = []
    
    print(f"🔬 Iniciando Benchmark Científico...")
    
    for n, noise, err_type in scenarios:
        df = injector.generate_dataset(gabarito, n, noise, err_type)
        
        for method_name, func in methods.items():
            start = time.perf_counter()
            predicted = func(df['Base_Suja'])
            duration = time.perf_counter() - start
            
            # Cálculo simplificado para o benchmark
            tp = ((df['is_dirty']) & (predicted == df['Gabarito'])).sum()
            fp = ((~df['is_dirty']) & (predicted != df['Gabarito'])).sum()
            fn = ((df['is_dirty']) & (predicted != df['Gabarito'])).sum()
            tn = ((~df['is_dirty']) & (predicted == df['Gabarito'])).sum()
            
            correction_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            damage_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results.append({
                'Tamanho': n, 'Ruido': noise, 'Tipo_Erro': err_type, 'Metodo': method_name,
                'Acuracia': accuracy_score(df['Gabarito'], predicted),
                'Taxa_Correcao': correction_rate, 'Taxa_Dano': damage_rate,
                'Tempo_ms': (duration / n) * 1000
            })

    return pd.DataFrame(results), gabarito, injector, cleaner

# ==============================================================================
# 4. FUNÇÃO NOVA: EXPORTAR VISUALIZAÇÃO (Base Suja vs Limpa)
# ==============================================================================
def exportar_amostra_visual(gabarito, injector, cleaner):
    print("\n📸 Gerando 'Fotografia' do processo (debug_comparativo.csv)...")
    
    # Criamos um cenário "casca grossa": Erros mistos e muito ruído (80%)
    df_visual = injector.generate_dataset(gabarito, size=100, noise_level=0.8, error_type='mixed')
    
    # Aplica os 3 métodos e cria colunas novas
    print("   -> Aplicando Frequência...")
    df_visual['Limpo_Frequencia'] = cleaner.metodo_frequencia(df_visual['Base_Suja'])
    
    print("   -> Aplicando Dicionário...")
    df_visual['Limpo_Dicionario'] = cleaner.metodo_dicionario(df_visual['Base_Suja'])
    
    print("   -> Aplicando Híbrido...")
    df_visual['Limpo_Hibrido'] = cleaner.metodo_hibrido(df_visual['Base_Suja'])
    
    # Verifica acertos (Booleano) para facilitar leitura
    df_visual['Acertou_Hibrido'] = df_visual['Limpo_Hibrido'] == df_visual['Gabarito']
    
    # Reordena colunas para facilitar leitura
    cols = ['Gabarito', 'Base_Suja', 'Limpo_Frequencia', 'Limpo_Dicionario', 'Limpo_Hibrido', 'Acertou_Hibrido']
    df_visual = df_visual[cols]
    
    # Salva
    df_visual.to_csv('debug_comparativo.csv', index=False, sep=';', encoding='utf-8-sig')
    print("✅ Arquivo 'debug_comparativo.csv' salvo com sucesso!")
    return df_visual

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    # 1. Roda Estatísticas
    df_results, gabarito, injector, cleaner = run_benchmark()
    
    summary = df_results.groupby('Metodo')[['Acuracia', 'Taxa_Correcao', 'Taxa_Dano']].mean()
    print("\n--- RESUMO ESTATÍSTICO ---")
    print(summary.sort_values('Acuracia', ascending=False))
    
    # 2. Gera Arquivo Visual (Base Suja vs Limpa)
    df_amostra = exportar_amostra_visual(gabarito, injector, cleaner)
    
    # Mostra uma prévia no terminal
    print("\n--- PREVIA DOS DADOS (Primeiras 5 linhas) ---")
    print(df_amostra[['Gabarito', 'Base_Suja', 'Limpo_Hibrido']].head().to_string())