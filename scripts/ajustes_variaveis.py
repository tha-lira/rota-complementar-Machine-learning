# ----------------------------------------------------
# ✨ Bibliotecas Importadas 
# ----------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

if 'Over18' in df.columns:
    df.drop('Over18', axis=1, inplace=True)

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

import os
import seaborn as sns
import matplotlib.pyplot as plt

# Criar pasta de saída, se ainda não existir
os.makedirs("figuras", exist_ok=True)

# Dicionário com colunas e seus respectivos insights
analises = {
    'MonthlyIncome': 'Funcionários com salários mais baixos tendem a sair mais?',
    'TotalWorkingYears': 'Profissionais com pouca experiência geral têm maior rotatividade?',
    'JobRole': 'Alguns cargos são mais críticos (ex: vendas, call center)?',
    'DistanceFromHome': 'Distância afeta a retenção?',
    'BusinessTravel': 'Viagens frequentes aumentam a saída?'
}

# Loop para gerar e salvar os gráficos
for coluna, insight in analises.items():
    plt.figure(figsize=(8, 4))

    if df[coluna].dtype == 'object' or str(df[coluna].dtype) == 'category':
        # Gráfico de barras com taxa média de Attrition para variáveis categóricas
        taxa_attrition = df.groupby(coluna)['Attrition'].mean().sort_values(ascending=False)
        sns.barplot(x=taxa_attrition.index, y=taxa_attrition.values, palette='coolwarm')
        plt.title(f'{coluna} vs Attrition\n💡 Insight: {insight}')
        plt.ylabel('Taxa média de rotatividade')
        plt.xticks(rotation=45, ha='right')
    else:
        # Gráfico de boxplot para variáveis numéricas
        sns.boxplot(x='Attrition', y=coluna, data=df, palette='Set2')
        plt.title(f'{coluna} vs Attrition\n💡 Insight: {insight}')
        plt.xlabel('Attrition (0 = Não, 1 = Sim)')
        plt.ylabel(coluna)

    # Salvar o gráfico
    filename = f"figuras/attrition_vs_{coluna.lower()}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
