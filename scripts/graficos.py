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
# 4. CÓPIA PARA VISUALIZAÇÃO (antes da codificação)
# ----------------------------------------------------

df_vis = df.copy()  # Usado apenas para visualização com labels legíveis

# ----------------------------------------------------
# 5. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
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
# 6. VERIFICAÇÃO FINAL DO DATAFRAME
# ----------------------------------------------------

print("\n--- ✅ DATAFRAME FINALMENTE LIMPO E CODIFICADO (df.info()) ---")
df.info()

# ----------------------------------------------------
# 7. GERAÇÃO DE GRÁFICOS COM INSIGHTS DE ROTATIVIDADE
# ----------------------------------------------------

# Garantir que 'Attrition' está numérica para cálculo
df_vis['Attrition'] = pd.to_numeric(df_vis['Attrition'], errors='coerce')

# Criar pasta se não existir
os.makedirs("figuras", exist_ok=True)

# Dicionário com colunas e perguntas de negócio
analises = {
    'MonthlyIncome': 'Funcionários com salários mais baixos tendem a sair mais?',
    'TotalWorkingYears': 'Profissionais com pouca experiência geral têm maior rotatividade?',
    'JobRole': 'Alguns cargos são mais críticos (ex: vendas, call center)?',
    'DistanceFromHome': 'Distância afeta a retenção?',
    'BusinessTravel': 'Viagens frequentes aumentam a saída?'
}

for coluna, insight in analises.items():
    try:
        plt.figure(figsize=(8, 4))

        if df_vis[coluna].dtype == 'object' or str(df_vis[coluna].dtype) == 'category':
            # Gráfico para variáveis categóricas
            taxa_attrition = df_vis.groupby(coluna)['Attrition'].mean(numeric_only=True).sort_values(ascending=False)
            sns.barplot(x=taxa_attrition.index, y=taxa_attrition.values, palette='coolwarm')
            plt.title(f'{coluna} vs Attrition\n💡 Insight: {insight}')
            plt.ylabel('Taxa média de rotatividade')
            plt.xlabel(coluna)
            plt.xticks(rotation=45, ha='right')

        else:
            # Verifica se existem dados suficientes em cada grupo de Attrition
            grupos = df_vis.groupby('Attrition')[coluna]
            if len(grupos) == 2 and all(len(g) > 0 for g in grupos):
                sns.boxplot(x='Attrition', y=coluna, data=df_vis)
                plt.title(f'{coluna} vs Attrition\n Insight: {insight}')
                plt.xlabel('Attrition (0 = Não, 1 = Sim)')
                plt.ylabel(coluna)
            else:
                print(f"⚠️ Dados insuficientes para plotar '{coluna}' com relação à Attrition.")
                plt.close()
                continue

        # Salvar gráfico
        filename = f"figuras/attrition_vs_{coluna.lower()}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Gráfico salvo: {filename}")

    except Exception as e:
        print(f"❌ Erro ao gerar gráfico para '{coluna}': {e}")
