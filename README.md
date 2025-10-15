# 📊 Projeto: Previsão de Rotatividade de Funcionários com Machine Learning

## 📁 1. Contexto do Projeto

No mercado atual, **reter talentos é uma vantagem estratégica**.  
A **rotatividade de funcionários (attrition)** impacta diretamente os custos de recrutamento, treinamento e produtividade das equipes.  

Este projeto tem como objetivo **prever a probabilidade de desligamento de um funcionário** com base em variáveis demográficas e comportamentais do RH, permitindo que a empresa adote **ações preventivas de retenção**.

---

## 🧠 2. Objetivo

Construir um **modelo supervisionado de Machine Learning** capaz de prever se um funcionário deixará ou não a empresa (problema de **classificação binária**), avaliando o desempenho de diferentes algoritmos e identificando os principais fatores associados à rotatividade.

---

## 🧰 3. Ferramentas e Tecnologias

- **Ambiente:** Google Colab / VS Code  
- **Linguagem:** Python  
- **Principais Bibliotecas:**
  - Manipulação e análise: `pandas`, `numpy`
  - Visualização: `matplotlib`, `seaborn`
  - Modelagem: `scikit-learn`, `xgboost`
  - Métricas e avaliação: `classification_report`, `confusion_matrix`, `roc_auc_score`

---

## 🧹 4. Preparação e Limpeza dos Dados

1. **Tratamento de valores nulos:**  
   - Colunas `NumCompaniesWorked` e `TotalWorkingYears` tiveram valores ausentes imputados com a mediana.

2. **Remoção de colunas redundantes:**  
   - Foram excluídas variáveis com valores constantes ou irrelevantes (`EmployeeCount`, `StandardHours`,  `EmployeeID`, `Over18`).

3. **Criação de novas variáveis (Feature Engineering):**  
   - `AgeGroup`: classificação etária (Jovem, Adulto, Sênior)  
   - `DistanceCategory`: categorização da distância casa–trabalho  
   - `ManyCompaniesWorked`: indicador binário para experiência em mais de 3 empresas  

4. **Codificação de variáveis categóricas:**  
   - Aplicado `LabelEncoder` para transformar variáveis do tipo texto em valores numéricos.

---

## 📈 5. Análise Exploratória (EDA)

Principais insights:

- **Idade:** Funcionários mais jovens apresentaram maior propensão a sair.  
- **Tempo de empresa:** A maior taxa de rotatividade ocorre nos **primeiros 2 anos de vínculo** (≈29%).  
- **Faixa salarial:** Funcionários com **salários mais baixos** têm maior tendência ao desligamento.  
- **Tempo com o gerente:** Funcionários com **menos tempo de gestão direta** apresentam maior rotatividade.

Gráficos desenvolvidos:
- Distribuição de idade média por status de desligamento  
- Taxa de rotatividade por tempo de empresa  
- Proporção de rotatividade por faixa etária  
- Correlação das variáveis com o `Attrition`  
- Importância das variáveis no modelo final  
- Matriz de confusão dos modelos

---

## 🤖 6. Modelagem Preditiva

Três algoritmos supervisionados foram avaliados:

| Modelo                 | Acurácia | AUC-ROC | Observações |
|-------------------------|-----------|----------|--------------|
| **Regressão Logística** | 62.7%     | 0.69     | Baixa capacidade de generalização |
| **XGBoost**             | 95.6%     | 0.99     | Excelente desempenho, mas leve overfitting |
| **Random Forest**       | 🥇 **97.2%** | **0.995** | Melhor equilíbrio entre precisão e robustez |

---

## 🔍 7. Interpretação dos Resultados

O modelo **Random Forest** apresentou o melhor desempenho geral, demonstrando:

- **Alta acurácia (97%)** e **excelente capacidade discriminativa (AUC 0.995)**  
- Robustez a **outliers**, pois utiliza regras de decisão em vez de médias  
- Boa capacidade de interpretação das variáveis mais relevantes:

| Variável | Importância (%) |
|-----------|------------------|
| Idade (`Age`) | 29.97 |
| Anos de trabalho total (`TotalWorkingYears`) | 25.96 |
| Número de empresas anteriores (`NumCompaniesWorked`) | 18.52 |
| Tempo com o gerente atual (`YearsWithCurrManager`) | 17.22 |
| Estado civil (`MaritalStatus`) | 8.33 |

---

## ⚠️ 8. Considerações e Limitações
 
- A ausência de remoção de outliers **não compromete a performance**, dado que o modelo de árvore é robusto, mas pode influenciar a **interpretabilidade dos resultados**.

---

## 🚀 9. Próximos Passos

- Implementar **validação cruzada (k-fold)** para melhor estimativa de performance.  
- Testar **modelos de interpretabilidade**, como SHAP ou LIME.  
- Criar um **dashboard interativo (Power BI ou Looker Studio)** para visualização das métricas de rotatividade.

---

## 📚 10. Estrutura do Projeto

