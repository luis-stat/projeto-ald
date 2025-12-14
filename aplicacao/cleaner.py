import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from rapidfuzz import fuzz, process
import unicodedata
import re

class DataCleaner:
    def __init__(self):
        self.cleaning_report = {}
        
    def normalize_text(self, text: Any) -> str:
        """
        Normalização profunda: Latin-ASCII, Lowercase e remoção de caracteres especiais.
        """
        if pd.isna(text): return ""
        text = str(text)
        # Normalização Unicode (remove acentos: ã -> a)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = text.lower().strip()
        # Mantém apenas letras, números e espaços
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        # Remove espaços duplos
        return re.sub(r'\s+', ' ', text)

    def get_tokens(self, text: str) -> Set[str]:
        """Quebra o texto normalizado em um conjunto de palavras (tokens)."""
        return set(text.split())
        
    def to_title_case_br(self, text: Any) -> Any:
        """Formata para Title Case respeitando preposições."""
        if pd.isna(text): return text
        text = str(text).strip()
        if not text: return text
        
        small_words = {'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'para', 'com'}
        words = text.split()
        title_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                title_words.append(word.capitalize())
            else:
                title_words.append(word.lower())
        return ' '.join(title_words)
        
    def standardize_date(self, date_str: Any) -> Any:
        """Padroniza datas para dd/mm/aaaa."""
        if pd.isna(date_str): return date_str
        str_date = str(date_str).strip()
        if not str_date: return np.nan
        try:
            date_obj = pd.to_datetime(str_date, dayfirst=True, errors='coerce')
            return date_obj.strftime('%d/%m/%Y') if not pd.isna(date_obj) else np.nan
        except: return np.nan

    def hierarchical_matcher(self, query: str, candidates_data: List[Dict], threshold: float) -> Optional[str]:
        """
        MÉTODO HÍBRIDO (HIERÁRQUICO):
        1. Normalização
        2. Filtro por Token Overlap
        3. Desempate por Fuzzy Token Sort Ratio
        """
        q_norm = self.normalize_text(query)
        if not q_norm: return None
        
        q_tokens = self.get_tokens(q_norm)
        
        # 1. Busca Exata
        for cand in candidates_data:
            if cand['norm'] == q_norm: return cand['orig']
                
        # 2. Filtro por Token Overlap
        best_overlap = 0
        overlap_candidates = []
        
        if q_tokens:
            for cand in candidates_data:
                if not cand['tokens']: continue
                intersection = len(q_tokens.intersection(cand['tokens']))
                
                if intersection > best_overlap:
                    best_overlap = intersection
                    overlap_candidates = [cand]
                elif intersection == best_overlap and intersection > 0:
                    overlap_candidates.append(cand)
        
        final_candidates = overlap_candidates if best_overlap > 0 else candidates_data
        candidate_strings = [c['norm'] for c in final_candidates]
        
        # 3. Fuzzy Sort
        if not candidate_strings: return None
        
        match_tuple = process.extractOne(q_norm, candidate_strings, scorer=fuzz.token_sort_ratio)
        
        if match_tuple:
            _, score, idx = match_tuple
            if score >= threshold:
                return final_candidates[idx]['orig']
            
        return None

    def hybrid_correction(self, series: pd.Series, reference_values: List[str], threshold: float = 85) -> Tuple[pd.Series, Dict]:
        """Aplica a correção híbrida."""
        series_clean = series.copy()
        unique_inputs = series.dropna().astype(str).unique()
        
        candidates_data = []
        for ref in reference_values:
            norm = self.normalize_text(str(ref))
            candidates_data.append({
                'orig': str(ref),
                'norm': norm,
                'tokens': self.get_tokens(norm)
            })
            
        corrections = {}
        rows_corrected = 0
        correction_cache = {}
        
        for val in unique_inputs:
            val_str = str(val)
            if val_str in reference_values: continue
            
            if val_str not in correction_cache:
                correction_cache[val_str] = self.hierarchical_matcher(val_str, candidates_data, threshold)
            
            match = correction_cache[val_str]
            
            if match and match != val_str:
                corrections[val_str] = match
        
        if corrections:
            series_clean = series_clean.replace(corrections)
            rows_corrected = sum(series.isin(corrections.keys()))
            
        return series_clean, {'rows_corrected': rows_corrected, 'corrections_made': corrections}

    def fallback_auto_correction(self, series: pd.Series, threshold: float = 85) -> Tuple[pd.Series, Dict]:
        """Método de Frequência (Fallback)."""
        counts = series.value_counts()
        total = len(series)
        if total == 0: return series, {}
        
        limit = max(5, total * 0.01)
        frequent_values = counts[counts >= limit].index.tolist()
        
        if not frequent_values: return series, {}
        
        return self.hybrid_correction(series, frequent_values, threshold)

    def clean_dataset(self, df: pd.DataFrame, type_predictions: Dict[str, str], 
                      reference_dictionaries: Dict[str, List[str]] = {}, 
                      similarity_threshold: float = 85) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        cleaned_df = df.copy()
        report = {'columns_cleaned': {}, 'total_rows_corrected': 0, 'total_rows_removed': 0}
        
        for col in df.columns:
            if col not in type_predictions: continue
            
            col_type = type_predictions[col]
            # CORREÇÃO: Inicializa 'rows_removed' com 0 para evitar KeyError
            stats = {'rows_corrected': 0, 'rows_removed': 0, 'method': 'none'}
            
            # 1. Tratamento de DATA_HORA
            if col_type == 'DATA_HORA':
                original_na = cleaned_df[col].isna().sum()
                cleaned_df[col] = cleaned_df[col].apply(self.standardize_date)
                stats['rows_removed'] = int(cleaned_df[col].isna().sum() - original_na)
                stats['method'] = 'date_standardization'
                
            # 2. Tratamento de TEXTO/CATEGORICO
            elif col_type in ['TEXTO_LIVRE', 'CATEGORICO_NOMINAL', 'CATEGORICO_ORDINAL', 'CATEGORICO_ESTADO', 'ID']:
                cleaned_df[col] = cleaned_df[col].apply(self.to_title_case_br)
                
                if col in reference_dictionaries and reference_dictionaries[col]:
                    cleaned_df[col], h_stats = self.hybrid_correction(
                        cleaned_df[col], 
                        reference_dictionaries[col], 
                        similarity_threshold
                    )
                    stats.update(h_stats)
                    stats['method'] = 'hybrid_dictionary (gabarito)'
                else:
                    if 'CATEGORICO' in col_type:
                        cleaned_df[col], f_stats = self.fallback_auto_correction(
                            cleaned_df[col], 
                            similarity_threshold
                        )
                        stats.update(f_stats)
                        stats['method'] = 'frequency_fallback (auto)'

            report['columns_cleaned'][col] = stats
            report['total_rows_corrected'] += stats.get('rows_corrected', 0)
            report['total_rows_removed'] += stats.get('rows_removed', 0)
                
        return cleaned_df, report