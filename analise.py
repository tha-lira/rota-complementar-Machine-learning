import os
import pandas as pd

# Define o caminho da pasta onde está o arquivo
pasta = r'C:\Users\Thais Lira\Documents\rota-complementar-Machine-learning'

# Junta o caminho da pasta com o nome do arquivo
caminho_arquivo = os.path.join(pasta, 'rh_data.csv')

df = pd.read_csv(caminho_arquivo)

print("Primeiras 5 linhas da tabela:")
print(df.head())
