import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

#----------------------------------------------------------------------------

# Codificação simples para análises
le = LabelEncoder()
categorical_columns = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'Attrition',
    'AgeGroup', 'DistanceCategory'
]

for col in categorical_columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 6.1 Média por Attrition
media_por_attrition = df.groupby('Attrition')[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']].mean().round(2)
print("\n📊 Média das variáveis numéricas por Attrition:")
print(media_por_attrition)

# 6.2 Correlação com Attrition
correlacoes = df.corr(numeric_only=True)['Attrition'].sort_values(ascending=False).round(2)
print("\n🔗 Correlação das variáveis com Attrition:")
print(correlacoes)

# 6.3 Rotatividade por tempo de empresa (faixas)
df['YearsBin'] = pd.cut(df['YearsAtCompany'], bins=[0, 2, 5, 10, 20, 40], labels=['0-2', '3-5', '6-10', '11-20', '21+'])
rotatividade_tempo = df.groupby('YearsBin')['Attrition'].agg(['count', 'sum'])
rotatividade_tempo['Taxa (%)'] = (rotatividade_tempo['sum'] / rotatividade_tempo['count'] * 100).round(2)
rotatividade_tempo.columns = ['Total', 'Desligados', 'Taxa (%)']
print("\n⏳ Rotatividade por Tempo de Empresa:")
print(rotatividade_tempo)

# 6.4 Proporção de Attrition por Faixa Etária
tabela_cruzada = pd.crosstab(df['AgeGroup'], df['Attrition'], normalize='index') * 100
tabela_cruzada.columns = ['Ficaram (%)', 'Saíram (%)']
print("\n📈 Proporção de Attrition por Faixa Etária:")
print(tabela_cruzada.round(2))

