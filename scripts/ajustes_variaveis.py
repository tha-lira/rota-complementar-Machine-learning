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
# 5. DIVISÃO DOS DADOS EM TREINO E TESTE (80/20)
# ----------------------------------------------------

X = df.drop('Attrition', axis=1) 
y = df['Attrition'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42, 
    stratify=y          
)

# ----------------------------------------------------
# 6. VERIFICAÇÃO DA DIVISÃO
# ----------------------------------------------------

print("\n--- Verificação da Proporção da Variável Alvo (Attrition) ---")
print("📊 Proporção geral:")
print(y.value_counts(normalize=True))
print("\n📊 Proporção no conjunto de teste:")
print(y_test.value_counts(normalize=True))

# ----------------------------------------------------
# 7. ANÁLISE DESCRITIVA E DE DISTRIBUIÇÃO
# ----------------------------------------------------

print("\n--- 📊 7.1 Medidas de Tendência Central e Dispersão (Variáveis Numéricas) ---")
print(df.describe().T) 

# ✅ Salvar gráficos de distribuição
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Age'], kde=True, bins=20, color='skyblue')
plt.title('Distribuição de Idade')

plt.subplot(1, 2, 2)
sns.histplot(df['MonthlyIncome'], kde=True, bins=30, color='lightcoral')
plt.title('Distribuição de Renda Mensal')

plt.tight_layout()
plt.savefig("figuras/distribuicao_idade_renda.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------------------------
# 8. AGRUPAMENTO E VISUALIZAÇÃO DE VARIÁVEIS CATEGÓRICAS
# ----------------------------------------------------

print("\n--- 📉 8.1 Taxa de Rotatividade (Attrition) por Categoria ---")

categorical_cols_to_analyze = [
    'Gender', 'MaritalStatus', 'JobRole', 'Department', 'AgeGroup' 
]

for col in categorical_cols_to_analyze:
    attrition_rate = df.groupby(col)['Attrition'].mean().sort_values(ascending=False)
    print(f"\nTaxa de Rotatividade por {col} (0 a 1):")
    print(attrition_rate)

    plt.figure(figsize=(10, 4))
    sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette="coolwarm")
    plt.title(f'Taxa Média de Attrition por {col}')
    plt.ylabel('Taxa de Attrition (Média)')
    plt.xlabel(col)
    plt.xticks(rotation=45, ha='right')

    # ✅ Salvar cada gráfico com nome baseado na coluna
    filename = f"figuras/rotatividade_{col.lower()}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
