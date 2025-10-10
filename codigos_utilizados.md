"""
⚙️ Funções e Códigos Utilizados no Projeto Spotify
Autor: Thaís Lira Apolinário
Descrição: Este arquivo reúne funções e códigos reutilizáveis
para limpeza, transformação e visualização dos dados.
"""

✨ Bibliotecas Importadas 

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
```

# ----------------------------------------------------

# 0. CARREGAR DADOS

```
df = pd.read_csv(r'C:\Users\Thais Lira\Documents\rota-complementar-Machine-learning\rh_data.csv')
```

# ----------------------------------------------------

1. TRATAMENTO DE NULOS (IMPUTAÇÃO DE MEDIANA)

```
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(df['NumCompaniesWorked'].median())
df['TotalWorkingYears'] = df['TotalWorkingYears'].fillna(df['TotalWorkingYears'].median())
print("Verificação de Nulos após Preenchimento:")
print(df[['NumCompaniesWorked', 'TotalWorkingYears']].isnull().sum())
```

# ----------------------------------------------------

2. REMOÇÃO DE COLUNAS

```
if 'Over18' in df.columns:
    df.drop('Over18', axis=1, inplace=True)
if 'EmployeeCount' in df.columns: 
    df.drop('EmployeeCount', axis=1, inplace=True)
```

# ----------------------------------------------------

3. CRIAÇÃO DE NOVAS VARIÁVEIS (FEATURE ENGINEERING)

```
def faixa_idade(idade):
    if idade <= 30:
        return 'Jovem'
    elif idade <= 45:
        return 'Adulto'
    else:
        return 'Sênior'
```

*Criação das colunas de texto*

```
df['AgeGroup'] = df['Age'].apply(faixa_idade)
dist_bins = [0, 5, 15, df['DistanceFromHome'].max() + 1] 
dist_labels = ['Perto', 'Médio', 'Longe']
df['DistanceCategory'] = pd.cut(df['DistanceFromHome'], bins=dist_bins, labels=dist_labels, include_lowest=True)
df['ManyCompaniesWorked'] = (df['NumCompaniesWorked'] > 3).astype(int)
```

*Visualização da criação*

```
print("\nNovas Variáveis Criadas (Amostra):")
print(df[['Age', 'AgeGroup', 'DistanceFromHome', 'DistanceCategory', 'NumCompaniesWorked', 'ManyCompaniesWorked']].sample(10))
```

# ----------------------------------------------------

4. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS (Label Encoding)

```
le = LabelEncoder()
categorical_columns = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'Attrition',
    'AgeGroup', 
    'DistanceCategory'
]
```

*Codificação das colunas existentes (texto para número)*

```
for col in categorical_columns:
    # A codificação direta funciona para colunas 'object' ou 'category'
    if df[col].dtype == 'object' or df[col].dtype == 'category':
        df[col] = le.fit_transform(df[col])
```

# ----------------------------------------------------


5. DIVISÃO DOS DADOS EM TREINO E TESTE (80/20)

*Separar variáveis independentes (X) e alvo (y)*

```
X = df.drop('Attrition', axis=1) 
y = df['Attrition'] # Variável alvo já codificada em números (0 e 1)
```

*Dividir a base em treino (80%) e teste (20%), mantendo a proporção da variável-alvo*

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,    # Garante resultados reprodutíveis
    stratify=y          # Mantém a proporção de Attrition (0 e 1) em ambos os conjuntos
)
```

# ----------------------------------------------------

6. VERIFICAÇÃO DA DIVISÃO

```
print("\n--- Verificação da Proporção da Variável Alvo (Attrition) ---")
print("📊 Proporção geral:")
print(y.value_counts(normalize=True))

print("\n📊 Proporção no conjunto de treino:")
print(y_train.value_counts(normalize=True))

print("\n📊 Proporção no conjunto de teste:")
print(y_test.value_counts(normalize=True))
```
