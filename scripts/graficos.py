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

# ----------------------------------------------------
# 📊 Visualizar somente as taxas de rotatividade (sem gráficos)
# ----------------------------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Carregar base
df = pd.read_csv(r'C:\Users\Thais Lira\Documents\rota-complementar-Machine-learning\dados\rh_data.csv')

# Tratamento de nulos
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(df['NumCompaniesWorked'].median())
df['TotalWorkingYears'] = df['TotalWorkingYears'].fillna(df['TotalWorkingYears'].median())

# Remover colunas redundantes
for col in ['Over18', 'EmployeeCount', 'StandardHours']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Criar novas variáveis
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

# Codificar variáveis categóricas
le = LabelEncoder()
categorical_columns = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'Attrition', 'AgeGroup', 'DistanceCategory'
]
for col in categorical_columns:
    if df[col].dtype in ['object', 'category']:
        df[col] = le.fit_transform(df[col])

# ----------------------------------------------------
# 1️⃣ Rotatividade por tempo com o gerente
# ----------------------------------------------------
rot_gerente = df.groupby('YearsWithCurrManager')['Attrition'].mean().sort_values(ascending=False)
print("\n📊 Taxa média de rotatividade por tempo com o gerente:")
print(rot_gerente)

# ----------------------------------------------------
# 2️⃣ Rotatividade por tempo de empresa
# ----------------------------------------------------
rot_empresa = df.groupby('YearsAtCompany')['Attrition'].mean().sort_values(ascending=False)
print("\n📊 Taxa média de rotatividade por tempo de empresa:")
print(rot_empresa)

# ----------------------------------------------------
# 3️⃣ Rotatividade por faixa salarial
# ----------------------------------------------------
# Criar faixas de salário
df['FaixaSalarial'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Baixa', 'Média-Baixa', 'Média-Alta', 'Alta'])
rot_salario = df.groupby('FaixaSalarial')['Attrition'].mean().sort_values(ascending=False)

print("\n📊 Taxa média de rotatividade por faixa salarial:")
print(rot_salario)

