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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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
# 5. DIVISÃO DA BASE (FEATURES E TARGET)
# ----------------------------------------------------
X = df.drop(['Attrition', 'EmployeeID'], axis=1)
y = df['Attrition']

# Divide em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- ✅ DIVISÃO TREINO/TESTE COMPLETA ---")

# ----------------------------------------------------
# 6. BALANCEAMENTO DE CLASSES (SMOTE)
# ----------------------------------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Treino Original (Não/Sim): {y_train.value_counts().values}")
print(f"Treino Balanceado (Não/Sim): {y_train_smote.value_counts().values}")

# ----------------------------------------------------
# 7. TREINAMENTO DO RANDOM FOREST
# ----------------------------------------------------
# Cria e treina o modelo Random Forest usando a base balanceada
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced')
rf_model.fit(X_train_smote, y_train_smote)

# ----------------------------------------------------
# 8. PREDIÇÃO NA BASE DE TESTE ORIGINAL
# ----------------------------------------------------
y_pred_rf = rf_model.predict(X_test)

# ----------------------------------------------------
# 9. CONFIRMAÇÃO DO DESEMPENHO E VARIÁVEIS
# ----------------------------------------------------

# 1. Relatório de Classificação (confirma Precision, Recall, F1)
print("\n--- 📊 RELATÓRIO DE CLASSIFICAÇÃO (RANDOM FOREST) ---")
print(classification_report(y_test, y_pred_rf))

# 2. Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Rotatividade (0)', 'Rotatividade (1)'],
            yticklabels=['Não Rotatividade (0)', 'Rotatividade (1)'])
plt.title('Matriz de Confusão - Random Forest')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

# 3. Importância das Variáveis
print("\n--- 🌟 IMPORTÂNCIA DAS VARIÁVEIS ---")
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_10_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh', color='darkgreen')
plt.title('Top 10 Importância de Variáveis - Random Forest')
plt.xlabel('Importância Relativa')
plt.gca().invert_yaxis()
plt.show()

print("\nTop 5 Variáveis Mais Importantes:")
print(top_10_features.head(5))

# ----------------------------------------------------
# 🔄 TREINAMENTO DE OUTROS MODELOS
# ----------------------------------------------------
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 🔹 1. Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_train_smote, y_train_smote)
y_pred_log = log_model.predict(X_test)

# 🔹 2. XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb_model.predict(X_test)

# ----------------------------------------------------
# 📊 FUNÇÃO PARA AVALIAÇÃO DOS MODELOS
# ----------------------------------------------------
def avaliar_modelo(nome, y_true, y_pred):
    print(f"\n📌 Resultados - {nome}")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1-Score :", round(f1_score(y_true, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_pred), 4))

# Avaliando todos os modelos
avaliar_modelo("Random Forest", y_test, y_pred_rf)
avaliar_modelo("Logistic Regression", y_test, y_pred_log)
avaliar_modelo("XGBoost", y_test, y_pred_xgb)
