# Relatório de Análise de Rotatividade (Attrition) – RH

## 1. Introdução

Este projeto visa analisar a rotatividade de colaboradores em uma empresa e desenvolver um modelo preditivo capaz de identificar os fatores que influenciam o desligamento, auxiliando o RH em ações de retenção.

## 2. Análise Exploratória de Dados (EDA)

### 2.1 Médias por Situação de Desligamento (`Attrition`)

| Variável            | Ficaram (0)    | Saíram (1)    |
| ------------------- | -------------- | ------------- |
| Idade média         | 37,56 anos     | 33,61 anos    |
| Renda mensal média  | R$ 65.672,00   | R$ 61.683,00  |
| TotalWorkingYears   | 11,86 anos     | 8,26 anos     |
| Tempo na empresa    | 7,37 anos      | 5,13 anos     |

💡 **Insights:**  
Funcionários mais jovens, com menor experiência e menor tempo de permanência na empresa apresentam maior propensão à rotatividade.

### 2.2 Correlação com a Rotatividade (`Attrition`)

| Variável             | Correlação  |
| -------------------- | ----------- |
| MaritalStatus        | +0,16       |
| YearsWithCurrManager | –0,16       |
| Age                  | –0,16       |
| TotalWorkingYears    | –0,17       |
| YearsAtCompany       | –0,13       |
| MonthlyIncome        | –0,03       |
| DistanceFromHome     | –0,01       |

💡 **Insights:**  
- Quanto menor a idade, experiência e tempo com o gerente atual, maior a chance de desligamento.  
- Funcionários solteiros (categoria de `MaritalStatus`) apresentam maior probabilidade de saída.

### 2.3 Rotatividade por Tempo de Empresa

| Tempo na Empresa | Total | Desligados | Taxa (%) |
| ---------------- | ----- | ---------- | -------- |
| 0–2 anos         | 894   | 258        | 28,86    |
| 3–5 anos         | 1.302 | 180        | 13,82    |
| 6–10 anos        | 1.344 | 165        | 12,28    |
| 11–20 anos       | 540   | 36         | 6,67     |
| 21+ anos         | 198   | 24         | 12,12    |

💡 **Insights:**  
Há alta rotatividade nos primeiros 2 anos, sugerindo necessidade de foco em integração e satisfação inicial.

### 2.4 Proporção de Rotatividade por Faixa Etária

| Faixa Etária | Ficaram (%) | Saíram (%) |
| ------------ | ----------- | ---------- |
| Jovem        | 87,30       | 12,70      |
| Adulto       | 74,09       | 25,91      |
| Sênior       | 87,55       | 12,45      |

💡 **Insights:**  
Adultos (31–45 anos) têm maior rotatividade, possivelmente devido a transições de carreira ou insatisfação.

## 3. Preparação e Tratamento de Dados

- Base utilizada: `rh_data.csv` com 4.410 colaboradores e 26 variáveis.  
- Variável alvo: `Attrition` (0 = Ficou, 1 = Saiu).  
- Tratamento de valores nulos: preenchimento de `NumCompaniesWorked` e `TotalWorkingYears` com mediana.  
- Engenharia de atributos: criação das variáveis `AgeGroup`, `DistanceCategory` e `ManyCompaniesWorked` (binária para >3 empresas anteriores).  
- Codificação: Label Encoding aplicado em variáveis categóricas.

## 4. Modelagem e Resultados

### 4.1 Algoritmos Utilizados

- Random Forest Classifier (modelo principal) com balanceamento via SMOTE.  
- Regressão Logística.  
- XGBoost (modelo alternativo).

### 4.2 Variáveis Utilizadas

`Age`, `TotalWorkingYears`, `MaritalStatus`, `YearsWithCurrManager`, `NumCompaniesWorked`.

### 4.3 Métricas de Avaliação

| Métrica           | Valor    |
| ----------------- | -------- |
| Acurácia          | 97,21%   |
| Recall (Classe 1) | 99%      |
| AUC-ROC           | 0,995    |

### 4.4 Matriz de Confusão

|           | Predito: Ficou | Predito: Saiu |
| --------- | -------------- | ------------- |
| Ficou (0) | 1.064          | 46            |
| Saiu (1)  | 16             | 1.094         |

💡 **Comentário:**  
O modelo apresenta baixa taxa de falsos negativos (16 casos), indicando boa capacidade para identificar quem irá sair.

### 4.5 Importância das Variáveis

| Variável             | Importância (%) |
| -------------------- | --------------- |
| Age                  | 29,97           |
| TotalWorkingYears    | 25,96           |
| NumCompaniesWorked   | 18,52           |
| YearsWithCurrManager | 17,22           |
| MaritalStatus        | 8,33            |

### 4.6 Comparação de Modelos

| Modelo              | Acurácia (%) | ROC-AUC |
| ------------------- | ------------ | ------- |
| Random Forest       | 97,21        | 0,995   |
| Regressão Logística | 97,21        | 0,995   |
| XGBoost             | 95,63        | 0,987   |

## 5. Conclusões

- O modelo preditivo apresentou alta precisão para identificar colaboradores com maior risco de desligamento.  
- Idade e experiência são os principais fatores preditores.  
- Funcionários mais jovens, com menor experiência e menor tempo de permanência na empresa apresentam maior propensão à rotatividade.  
- A rotatividade elevada nos primeiros dois anos indica necessidade de melhorias no onboarding e engajamento inicial.

## 6. Recomendações

- Monitorar especialmente os funcionários nos primeiros 2 anos.  
- Implementar programas de mentoria e engajamento para colaboradores jovens.  
- Avaliar a relação entre gestores e colaboradores para reduzir a rotatividade.  
- Desenvolver dashboards interativos (ex.: Streamlit) para uso contínuo do modelo preditivo.
