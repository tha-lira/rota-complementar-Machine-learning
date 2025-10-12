# --------------------
# 🔹 1. Importação de Bibliotecas
# --------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Definindo a SEED para reprodutibilidade
SEED = 42

# --------------------
# 🔹 2. Importação e Limpeza Inicial dos Dados
# --------------------
# Nota: Substitua pelo seu caminho de arquivo se não estiver no Colab/ambiente configurado
df = pd.read_csv('dados/rh_data.csv') 

# Tratamento de Nulos (Imputação de Mediana, robusta contra outliers)
df['NumCompaniesWorked'] = df['NumCompaniesWorked'].fillna(df['NumCompaniesWorked'].median())
df['TotalWorkingYears'] = df['TotalWorkingYears'].fillna(df['TotalWorkingYears'].median())

# Remoção de Colunas Inúteis/Constantes/ID
colunas_remover = [
    'EmployeeCount',    # Constante (valor 1)
    'StandardHours',    # Constante (valor 8)
    'Over18',           # Constante (valor 'Y')
    'EmployeeID'        # Identificador Único
]
df_clean = df.drop(colunas_remover, axis=1, errors='ignore')
print(f"Colunas removidas: {colunas_remover}")
print(f"Número de colunas após limpeza: {df_clean.shape[1]}")


# --------------------
# 🔹 3. Feature Engineering (Criação de Novas Variáveis)
# --------------------
def faixa_idade(idade):
    if idade <= 30:
        return 'Jovem'
    elif idade <= 45:
        return 'Adulto'
    else:
        return 'Sênior'

df_clean['AgeGroup'] = df_clean['Age'].apply(faixa_idade)

# Criação da Razão de Experiência na Empresa (novo indicador)
df_clean['ExperienceRatio'] = df_clean['YearsAtCompany'] / (df_clean['TotalWorkingYears'] + 1e-6)


# --------------------
# 🔹 4. Codificação de Variáveis Categóricas (Label Encoding)
# --------------------
le = LabelEncoder()
for col in df_clean.select_dtypes(include=['object', 'category']).columns:
    df_clean[col] = le.fit_transform(df_clean[col])


# --------------------
# 🔹 5. Preparação Final e Divisão em Treino/Teste
# --------------------

# Preparação do X final (Features)
# Remove Age e TotalWorkingYears originais, pois usamos suas versões transformadas (AgeGroup, ExperienceRatio)
X = df_clean.drop(['Attrition', 'Age', 'TotalWorkingYears'], axis=1)
y = df_clean['Attrition']

# Divisão Estratificada (mantém a proporção da variável resposta)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("\nDistribuição Attrition no Treino:", y_train.value_counts(normalize=True))


# --------------------
# 🔹 6. Seleção de Variáveis (Aplicada após a divisão)
# --------------------
# Seleciona as 15 melhores features com base na estatística F
selector = SelectKBest(score_func=f_classif, k=15)

# A seleção é feita apenas no conjunto de TREINO para evitar vazamento
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Obtendo os nomes das features selecionadas (para análise final)
feature_names = X.columns[selector.get_support()]
print("\nTop Features Selecionadas (K=15):", feature_names.tolist())


# --------------------
# 🔹 7. Treinamento e Comparação de Modelos (2.3.1)
# --------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
}

results = {}

for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Foco no F1-Score da Classe 1 (Attrition=Yes)
    f1_score_attrition = report['1']['f1-score']
    accuracy = report['accuracy']
    
    results[name] = {'Accuracy': accuracy, 'F1-Attrition': f1_score_attrition, 'Model': model}

    print(f"\n--- 🧠 Modelo: {name} ---")
    print(f"Accuracy Geral: {accuracy:.4f}")
    print(f"F1-Score (Attrition=Yes): {f1_score_attrition:.4f}")
    print(classification_report(y_test, y_pred))


# --------------------
# 🔹 8. Escolha e Análise do Modelo Final
# --------------------
best_model_name = max(results, key=lambda k: results[k]['F1-Attrition'])
best_model_info = results[best_model_name]
best_model = best_model_info['Model']

print("\n=======================================================")
print(f"🏆 MODELO VENCEDOR (Melhor F1-Score na Classe 1): {best_model_name}")
print(f"F1-Score (Attrition=Yes) no Teste: {best_model_info['F1-Attrition']:.4f}")
print("=======================================================")

# Se o modelo vencedor for baseado em árvores (Random Forest ou XGBoost), 
# extraímos as Feature Importances
if best_model_name in ["Random Forest", "XGBoost"]:
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        # Para modelos lineares, usamos coeficientes
        importances = np.abs(best_model.coef_[0])
    else:
        importances = None

    if importances is not None:
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\n--- 🌟 Top 10 Fatores de Rotatividade (Feature Importances) ---")
        print(feature_importance_df.head(10).to_string(index=False))

# Fim do escopo do projeto: Limpeza, Feature Engineering, Seleção, Treinamento e Comparação.