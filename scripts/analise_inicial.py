# ----------------------------------------------------
# ✨ Bibliotecas Importadas 
# ----------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------
# 0. CARREGAR DADOS
# ----------------------------------------------------

df = pd.read_csv(r'C:\Users\Thais Lira\Documents\rota-complementar-Machine-learning\dados\rh_data.csv')

# ----------------------------------------------------
# 1. IDENTIFICAÇÃO DE VALORES NULOS
# ----------------------------------------------------

print("\n🔍 Verificando valores nulos por coluna:\n")
print(df.isnull().sum())

# ----------------------------------------------------
# 2. IDENTIFICAÇÃO DE DADOS DUPLICADOS
# ----------------------------------------------------

duplicados = df.duplicated().sum()
print(f"\n📌 Total de linhas duplicadas: {duplicados}")

# ----------------------------------------------------
# 3. ANÁLISE DE VARIÁVEIS CATEGÓRICAS
# ----------------------------------------------------

# Identifica colunas do tipo 'object' (categóricas não codificadas)
categorical_cols = df.select_dtypes(include='object').columns

print("\n🔎 Valores únicos por coluna categórica:\n")
for col in categorical_cols:
    print(f"Coluna: {col}")
    print(df[col].unique())
    print("-" * 40)

# ----------------------------------------------------
# 4. ANÁLISE DE VARIÁVEIS NUMÉRICAS E DETECÇÃO DE OUTLIERS
# ----------------------------------------------------

# Seleciona colunas numéricas
numericas = df.select_dtypes(include=[np.number])

# Função para detectar outliers com dois métodos
def detectar_outliers(coluna):
    dados = numericas[coluna]

    # --- MÉTODO 1: Desvio Padrão (±3σ)
    media = dados.mean()
    desvio = dados.std()
    limite_inferior_std = media - 3 * desvio
    limite_superior_std = media + 3 * desvio
    outliers_std = dados[(dados < limite_inferior_std) | (dados > limite_superior_std)]

    # --- MÉTODO 2: IQR
    Q1 = dados.quantile(0.25)
    Q3 = dados.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior_iqr = Q1 - 1.5 * IQR
    limite_superior_iqr = Q3 + 1.5 * IQR
    outliers_iqr = dados[(dados < limite_inferior_iqr) | (dados > limite_superior_iqr)]

    # Resultado
    print(f"\n📌 Variável: {coluna}")
    print(f"  - Média: {media:.2f} | Desvio Padrão: {desvio:.2f}")
    print(f"  - Limites (±3σ): {limite_inferior_std:.2f} a {limite_superior_std:.2f}")
    print(f"  - Outliers (±3σ): {len(outliers_std)} valores")
    print(f"  - Limites (IQR): {limite_inferior_iqr:.2f} a {limite_superior_iqr:.2f}")
    print(f"  - Outliers (IQR): {len(outliers_iqr)} valores")

print("\n📉 Análise de Outliers nas variáveis numéricas:\n")
for coluna in numericas.columns:
    detectar_outliers(coluna)
