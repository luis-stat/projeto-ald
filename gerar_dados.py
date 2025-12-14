import pandas as pd
import numpy as np
import random

# Semente para reproduzibilidade
random.seed(42)
np.random.seed(42)

def sujar_texto(texto, nivel_erro=0.8):
    """Aplica erros de digitação, case, inversão ou remoção de tokens."""
    if pd.isna(texto) or random.random() > nivel_erro:
        return texto
    
    tipo_erro = random.choice(['typo', 'case', 'token', 'swap'])
    texto = str(texto)
    
    # Erro 1: Case (Caixa Alta/Baixa misturada)
    if tipo_erro == 'case':
        return ''.join([c.upper() if i%2==0 else c.lower() for i, c in enumerate(texto)])
    
    # Erro 2: Typo (Troca caracter)
    if tipo_erro == 'typo':
        if len(texto) < 2: return texto
        idx = random.randint(0, len(texto)-1)
        chars = list(texto)
        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return "".join(chars)
    
    # Erro 3: Perda de Token (Remove uma palavra)
    if tipo_erro == 'token':
        parts = texto.split()
        if len(parts) > 1:
            parts.pop(random.randint(0, len(parts)-1))
            return " ".join(parts)
        return texto
        
    # Erro 4: Inversão (São Paulo -> Paulo São)
    if tipo_erro == 'swap':
        parts = texto.split()
        if len(parts) >= 2:
            parts.reverse()
            return " ".join(parts)
        return texto

    return texto

def sujar_data(data_obj):
    """Gera datas em formatos variados ou inválidos."""
    if pd.isna(data_obj): return np.nan
    r = random.random()
    
    if r < 0.1: return np.nan # Dado faltante
    if r < 0.3: return data_obj.strftime("%Y-%m-%d") # Formato ISO
    if r < 0.5: return data_obj.strftime("%d-%m-%y") # Ano com 2 dígitos
    if r < 0.6: return data_obj.strftime("%d/%m/%Y").replace("/", ".") # Separador ponto
    if r < 0.7: return "Data Inválida" # Lixo
    return data_obj.strftime("%d/%m/%Y") # Correto

# --- DADOS GABARITO (Simulando o que existe na pasta de treinamento) ---
# O sistema deve reconhecer estas colunas e limpar perfeitamente.
bairros_corretos = [
    "Jardim Paulista", "Vila Madalena", "Copacabana", "Leblon", 
    "Centro Cívico", "Savassi", "Moinhos de Vento", "Asa Norte"
]

setores_corretos = [
    "Hortifruti", "Padaria e Confeitaria", "Açougue e Peixaria", 
    "Limpeza e Higiene", "Bebidas Alcoólicas", "Laticínios"
]

companhias_corretas = [
    "Latam Airlines", "Gol Linhas Aéreas", "Azul Linhas Aéreas", 
    "Tap Air Portugal", "American Airlines"
]

# Gerar 100 linhas
N = 100
df = pd.DataFrame({
    'id_transacao': range(1, N + 1),
    
    # Coluna 1: Bairro (Deve ativar Híbrido pois existe em imoveis_aluguel.csv)
    'bairro': [sujar_texto(random.choice(bairros_corretos)) for _ in range(N)],
    
    # Coluna 2: Setor (Deve ativar Híbrido pois existe em vendas_supermercado.csv)
    'setor': [sujar_texto(random.choice(setores_corretos)) for _ in range(N)],
    
    # Coluna 3: Companhia (Deve ativar Híbrido pois existe em viagens_aereas.csv)
    'companhia': [sujar_texto(random.choice(companhias_corretas)) for _ in range(N)],
    
    # Coluna 4: Data (Deve padronizar formato)
    'data_compra': [sujar_data(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)) for i in range(N)],
    
    # Coluna 5: Categoria desconhecida (Deve usar Frequência/Auto)
    'observacao_cliente': [sujar_texto(random.choice(['Entregar rápido', 'Embalagem presente', 'Ligar antes', 'Deixar na portaria'])) for _ in range(N)]
})

# Salvar
df.to_csv("base_teste_suja.csv", index=False)
print("✅ Arquivo 'base_teste_suja.csv' gerado com sucesso!")
print("Exemplo de sujeira gerada:")
print(df[['bairro', 'setor']].head())