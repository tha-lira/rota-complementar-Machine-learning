# ----------------------------------------------------
# ✨ Bibliotecas Importadas 
# ----------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------
# 0. CARREGAR DADOS
# ----------------------------------------------------

df = pd.read_csv(r'C:\Users\Thais Lira\Documents\rota-complementar-Machine-learning\dados\rh_data.csv')

# ----------------------------------------------------
# 1. TRATAMENTO DE NULOS (IMPUTAÇÃO DE MEDIANA)
# ----------------------------------------------------

df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(df['NumCompaniesWorked'].median())
df['TotalWorkingYears'] = df['TotalWorkingYears'].fillna(df['TotalWorkingYears'].median())
print("Verificação de Nulos após Preenchimento:")
print(df[['NumCompaniesWorked', 'TotalWorkingYears']].isnull().sum())

# ----------------------------------------------------
# 2. REMOÇÃO DE COLUNAS
# ----------------------------------------------------

colunas_para_remover = ['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeID']
df.drop(columns=[col for col in colunas_para_remover if col in df.columns], inplace=True)

# ----------------------------------------------------
# 3. CRIAÇÃO DE NOVAS VARIÁVEIS (FEATURE ENGINEERING)
# ----------------------------------------------------

def faixa_idade(idade):
    if idade <= 30:
        return 'Jovem'
    elif idade <= 45:
        return 'Adulto'
    else:
        return 'Sênior'

df['AgeGroup'] = df['Age'].apply(faixa_idade)
dist_bins = [0, 5, 15, df['DistanceFromHome'].max() + 1] 
dist_labels = ['Perto', 'Médio', 'Longe']
df['DistanceCategory'] = pd.cut(df['DistanceFromHome'], bins=dist_bins, labels=dist_labels, include_lowest=True)
df['ManyCompaniesWorked'] = (df['NumCompaniesWorked'] > 3).astype(int)

print("\nNovas Variáveis Criadas (Amostra):")
print(df[['Age', 'AgeGroup', 'DistanceFromHome', 'DistanceCategory', 'NumCompaniesWorked', 'ManyCompaniesWorked']].sample(5))

# ----------------------------------------------------
# 4. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
# ----------------------------------------------------

le = LabelEncoder()
categorical_columns = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'Attrition',
    'AgeGroup', 
    'DistanceCategory'
]

for col in categorical_columns:
    if df[col].dtype in ['object', 'category']:
        df[col] = le.fit_transform(df[col])

# ----------------------------------------------------
# 4.5. VERIFICAÇÃO FINAL: DATAFRAME LIMPO E CODIFICADO
# ----------------------------------------------------

print("\n--- ✅ DATAFRAME FINALMENTE LIMPO E CODIFICADO (df.info()) ---")
df.info() 

#-------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configurações gerais do seaborn
sns.set(style="whitegrid")

# Caminho para salvar os gráficos
save_path = r'C:\Users\Thais Lira\Documents\rota-complementar-Machine-learning\figuras'

# Criar a pasta caso não exista
os.makedirs(save_path, exist_ok=True)

# 1. Idade média por situação de desligamento
media_idade = {'Ficaram': 37.56, 'Saíram': 33.61}
plt.figure(figsize=(6,4))
sns.barplot(x=list(media_idade.keys()), y=list(media_idade.values()), palette='muted')
plt.title('Idade Média por Situação de Desligamento')
plt.ylabel('Idade Média (anos)')
plt.xlabel('Situação')
plt.savefig(os.path.join(save_path, 'idade_media_situacao_desligamento.png'))
plt.close()

# 2. Correlação com Attrition (simplificado)
correlacoes = {
    'MaritalStatus': 0.16,
    'YearsWithCurrManager': -0.16,
    'Age': -0.16,
    'TotalWorkingYears': -0.17,
    'YearsAtCompany': -0.13,
    'MonthlyIncome': -0.03,
    'DistanceFromHome': -0.01
}
plt.figure(figsize=(8,5))
sns.barplot(x=list(correlacoes.keys()), y=list(correlacoes.values()), palette='coolwarm')
plt.title('Correlação das Variáveis com Rotatividade')
plt.ylabel('Correlação')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'correlacao_rotatividade.png'))
plt.close()

# 3. Rotatividade por tempo de empresa
tempo_empresa = ['0–2 anos', '3–5 anos', '6–10 anos', '11–20 anos', '21+ anos']
total = [894, 1302, 1344, 540, 198]
desligados = [258, 180, 165, 36, 24]
taxa = [28.86, 13.82, 12.28, 6.67, 12.12]

df_tempo = pd.DataFrame({
    'Tempo de Empresa': tempo_empresa,
    'Total': total,
    'Desligados': desligados,
    'Taxa (%)': taxa
})

plt.figure(figsize=(10,6))
sns.barplot(data=df_tempo, x='Tempo de Empresa', y='Taxa (%)', palette='magma')
plt.title('Taxa de Rotatividade por Tempo de Empresa')
plt.ylabel('Taxa de Rotatividade (%)')
plt.xlabel('Tempo de Empresa')
plt.savefig(os.path.join(save_path, 'taxa_rotatividade_tempo_empresa.png'))
plt.close()

# 4. Proporção de Rotatividade por Faixa Etária (barras empilhadas)
faixas = ['Jovem', 'Adulto', 'Sênior']
ficaram = [87.30, 74.09, 87.55]
sairam = [12.70, 25.91, 12.45]

df_faixa = pd.DataFrame({
    'Faixa Etária': faixas,
    'Ficaram (%)': ficaram,
    'Saíram (%)': sairam
})

plt.figure(figsize=(8,6))
plt.bar(df_faixa['Faixa Etária'], df_faixa['Ficaram (%)'], label='Ficaram', color='skyblue')
plt.bar(df_faixa['Faixa Etária'], df_faixa['Saíram (%)'], bottom=df_faixa['Ficaram (%)'], label='Saíram', color='salmon')
plt.title('Proporção de Rotatividade por Faixa Etária')
plt.ylabel('Percentual (%)')
plt.xlabel('Faixa Etária')
plt.legend()
plt.savefig(os.path.join(save_path, 'proporcao_rotatividade_faixa_etaria.png'))
plt.close()

# 5. Importância das Variáveis
variaveis = ['Age', 'TotalWorkingYears', 'NumCompaniesWorked', 'YearsWithCurrManager', 'MaritalStatus']
importancia = [29.97, 25.96, 18.52, 17.22, 8.33]

plt.figure(figsize=(8,5))
sns.barplot(x=importancia, y=variaveis, palette='viridis')
plt.title('Importância das Variáveis no Modelo Random Forest')
plt.xlabel('Importância (%)')
plt.ylabel('Variável')
plt.savefig(os.path.join(save_path, 'importancia_variaveis_random_forest.png'))
plt.close()

# 6. Matriz de Confusão
matriz_confusao = np.array([[1064, 46],
                           [16, 1094]])

plt.figure(figsize=(6,5))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=['Ficou', 'Saiu'], yticklabels=['Ficou', 'Saiu'])
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.savefig(os.path.join(save_path, 'matriz_confusao.png'))
plt.close()

print(f"Gráficos salvos na pasta: {save_path}")
