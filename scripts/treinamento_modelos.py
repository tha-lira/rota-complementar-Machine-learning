import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

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

# Codificar variáveis categóricas importantes
le = LabelEncoder()
for col in ['MaritalStatus']:
    df[col] = le.fit_transform(df[col])

# Variáveis selecionadas para previsão
features = ['Age', 'TotalWorkingYears', 'MaritalStatus', 'YearsWithCurrManager', 'NumCompaniesWorked']
X = df[features]
y = df['Attrition']

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Modelo Regressão Logística
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]

print("🔷 Regressão Logística:")
print(classification_report(y_test, y_pred_logreg))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_logreg))
print(f"Acurácia: {accuracy_score(y_test, y_pred_logreg)*100:.2f}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_logreg):.3f}\n")

# Modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("🔷 Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_rf))
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.3f}\n")

# Modelo XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("🔷 XGBoost:")
print(classification_report(y_test, y_pred_xgb))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_xgb))
print(f"Acurácia: {accuracy_score(y_test, y_pred_xgb)*100:.2f}%")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.3f}\n")

# Comparação resumida
resultados = [
    {"Modelo": "Regressão Logística", "Accuracy": accuracy_score(y_test, y_pred_logreg), "ROC_AUC": roc_auc_score(y_test, y_prob_logreg)},
    {"Modelo": "Random Forest", "Accuracy": accuracy_score(y_test, y_pred_rf), "ROC_AUC": roc_auc_score(y_test, y_prob_rf)},
    {"Modelo": "XGBoost", "Accuracy": accuracy_score(y_test, y_pred_xgb), "ROC_AUC": roc_auc_score(y_test, y_prob_xgb)},
]

df_resultados = pd.DataFrame(resultados)
print("📊 Comparação dos Modelos:")
print(df_resultados.sort_values(by='ROC_AUC', ascending=False))
