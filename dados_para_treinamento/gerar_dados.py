import pandas as pd
import numpy as np
import random
import os

# Configurações
OUTPUT_DIR = "." 
random.seed(2024)
np.random.seed(2024)

def random_dates(start, end, n):
    start_u = start.value//10**9
    end_u = end.value//10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

print("Gerando Lote 1")
n = 300 

# 1. AVIAÇÃO (viagens_aereas.csv)
df_av = pd.DataFrame({
    'flight_no': [f'VOO-{random.randint(100,999)}' for _ in range(n)],
    'companhia': np.random.choice(['Gol', 'Latam', 'Azul', 'TAP'], n),
    'origem': np.random.choice(['GRU', 'GIG', 'MIA', 'LIS'], n),
    'destino': np.random.choice(['JFK', 'LHR', 'CDG', 'DXB'], n),
    'atraso_minutos': np.random.randint(0, 180, n),
    'data_partida': random_dates(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'), n),
    'cancelado': np.random.choice([True, False], n, p=[0.05, 0.95])
})
df_av.to_csv('viagens_aereas.csv', index=False)

# 2. ENERGIA (consumo_inteligente.csv)
df_eng = pd.DataFrame({
    'smart_meter_id': [f'SM-{i:06d}' for i in range(n)],
    'consumo_kwh': np.round(np.random.uniform(50.0, 500.0, n), 2),
    'custo_estimado': np.round(np.random.uniform(20.0, 300.0, n), 2),
    'bandeira_tarifaria': np.random.choice(['Verde', 'Amarela', 'Vermelha 1', 'Vermelha 2'], n), # Ordinal
    'data_leitura': random_dates(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-06-01'), n),
    'bairro_unidade': np.random.choice(['Centro', 'Sul', 'Norte', 'Leste'], n)
})
df_eng.to_csv('consumo_inteligente.csv', index=False)

# 3. BIBLIOTECA (biblioteca_emprestimos.csv)
df_bib = pd.DataFrame({
    'isbn_livro': [f'978-{random.randint(100000000,999999999)}' for _ in range(n)],
    'titulo_livro': [f'Livro Genérico {i}' for i in range(n)],
    'genero_literario': np.random.choice(['Ficção', 'História', 'Ciência', 'Biografia'], n),
    'dias_atraso': np.random.randint(0, 30, n),
    'multa_devida': np.round(np.random.uniform(0, 50.0, n), 2),
    'data_devolucao': random_dates(pd.to_datetime('2023-01-01'), pd.to_datetime('2023-12-31'), n)
})
df_bib.to_csv('biblioteca_emprestimos.csv', index=False)

# 4. REDES SOCIAIS (social_posts.csv)
df_soc = pd.DataFrame({
    'post_id': [f'P-{i:05d}' for i in range(n)],
    'autor_handle': [f'@usuario_{random.randint(1,100)}' for _ in range(n)],
    'plataforma': np.random.choice(['Twitter', 'Instagram', 'LinkedIn', 'TikTok'], n),
    'likes_count': np.random.randint(0, 10000, n),
    'shares_count': np.random.randint(0, 500, n),
    'sentimento_ia': np.random.choice(['Positivo', 'Negativo', 'Neutro'], n),
    'data_publicacao': random_dates(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'), n)
})
df_soc.to_csv('social_posts.csv', index=False)

# 5. GAMES (player_stats.csv)
df_game = pd.DataFrame({
    'player_tag': [f'Player{i}#BR' for i in range(n)],
    'game_mode': np.random.choice(['Ranked', 'Casual', 'Tournament'], n),
    'kills': np.random.randint(0, 50, n),
    'deaths': np.random.randint(0, 50, n),
    'match_duration_sec': np.random.randint(300, 3600, n),
    'victory': np.random.choice([True, False], n),
    'rank_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'], n) # Ordinal
})
df_game.to_csv('player_stats.csv', index=False)

# 6. SEGUROS (sinistros_auto.csv)
df_seg = pd.DataFrame({
    'police_id': [f'POL-{i:06d}' for i in range(n)],
    'valor_sinistro': np.round(np.random.uniform(1000.0, 50000.0, n), 2),
    'tipo_incidente': np.random.choice(['Colisão', 'Roubo', 'Enchente', 'Terceiros'], n),
    'status_analise': np.random.choice(['Em Análise', 'Aprovado', 'Rejeitado', 'Pago'], n),
    'data_ocorrencia': random_dates(pd.to_datetime('2023-01-01'), pd.to_datetime('2023-12-31'), n),
    'franquia_paga': np.random.choice([True, False], n)
})
df_seg.to_csv('sinistros_auto.csv', index=False)

# 7. AGRICULTURA (colheita_safra.csv)
df_agro = pd.DataFrame({
    'talhao_id': [f'T-{random.randint(1,50)}' for _ in range(n)],
    'cultura': np.random.choice(['Soja', 'Milho', 'Trigo', 'Café'], n),
    'produtividade_ton': np.round(np.random.uniform(2.0, 10.0, n), 2),
    'umidade_solo': np.round(np.random.uniform(10.0, 40.0, n), 1),
    'data_colheita': random_dates(pd.to_datetime('2023-01-01'), pd.to_datetime('2023-12-31'), n),
    'qualidade_grao': np.random.choice(['Tipo A', 'Tipo B', 'Tipo C'], n) # Ordinal
})
df_agro.to_csv('colheita_safra.csv', index=False)

# 8. RECRUTAMENTO (candidatos_vagas.csv)
df_rec = pd.DataFrame({
    'candidato_id': [f'CAND-{i:04d}' for i in range(n)],
    'vaga_aplicada': np.random.choice(['Dev Junior', 'Dev Senior', 'Data Scientist', 'PM'], n),
    'anos_experiencia': np.random.randint(0, 15, n),
    'pretensao_salarial': np.round(np.random.uniform(3000.0, 20000.0, n), 2),
    'escolaridade': np.random.choice(['Médio', 'Superior', 'Pós-Graduação', 'Mestrado'], n), # Ordinal
    'contratado': np.random.choice([True, False], n, p=[0.1, 0.9])
})
df_rec.to_csv('candidatos_vagas.csv', index=False)

# 9. TRANSPORTE PÚBLICO (metro_fluxo.csv)
df_trans = pd.DataFrame({
    'estacao_id': np.random.choice(['Sé', 'Luz', 'Paulista', 'Pinheiros'], n),
    'horario_pico': np.random.choice([True, False], n),
    'qtd_passageiros': np.random.randint(50, 5000, n),
    'linha_cor': np.random.choice(['Azul', 'Verde', 'Amarela', 'Vermelha'], n),
    'tempo_espera_seg': np.random.randint(60, 600, n),
    'data_registro': random_dates(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-02-01'), n)
})
df_trans.to_csv('metro_fluxo.csv', index=False)

# 10. ACADEMIA (treinos_fitness.csv)
df_gym = pd.DataFrame({
    'membro_id': [f'GYM-{i:04d}' for i in range(n)],
    'tipo_treino': np.random.choice(['Musculação', 'Cardio', 'Crossfit', 'Pilates'], n),
    'calorias_queimadas': np.random.randint(100, 800, n),
    'duracao_minutos': np.random.randint(20, 90, n),
    'bpm_medio': np.random.randint(80, 160, n),
    'data_treino': random_dates(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-03-01'), n)
})
df_gym.to_csv('treinos_fitness.csv', index=False)

print("Arquivos do Lote 1 gerados com sucesso!")