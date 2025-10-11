# 🟥 2.3 Relatório da Técnica de Análise: Modelagem e Comparação

📌 **Objetivo:** Treinar e comparar diferentes modelos de **classificação supervisionada** para prever a **Rotatividade de Funcionários (Attrition)**, com foco na **classe minoritária (rotatividade = 1)**. O conjunto de dados foi previamente balanceado com **SMOTE** para mitigar o desbalanceamento.

---

### 🔴 2.3.1 Utilização de Aprendizado de Máquina Supervisionado

Três algoritmos foram treinados com os dados balanceados e testados em uma amostra representativa (desbalanceada), mantendo a distribuição real da rotatividade:

| Status da Base         | Classe 0 (Não) | Classe 1 (Sim - Rotatividade) | Observação                  |
|-----------------------|----------------|-------------------------------|-----------------------------|
| **Treino Original**    | 2.959          | 569                           | Base desbalanceada          |
| **Treino Balanceado**  | 2.959          | 2.959                         | Após aplicação de SMOTE     |

---

📊 **Comparação de Desempenho dos Modelos**

A métrica mais importante para este projeto é o **Recall da classe 1 (rotatividade)**, ou seja, a capacidade de **detectar corretamente os funcionários que sairão da empresa**.

| Modelo                | Acurácia | Recall (Rotatividade) | Precision (Rotatividade) | F1-Score | ROC-AUC |
|-----------------------|----------|----------------------|-------------------------|----------|---------|
| 🏆 **XGBoost (Vencedor)** | **0.9955** | **0.9718**            | **1.0000**                | **0.9857** | **0.9859** |
| Random Forest         | 0.9626   | 0.8803               | 0.8865                  | 0.8834   | 0.9293  |
| Regressão Logística   | 0.6837   | 0.5493               | 0.2662                  | 0.3586   | 0.6294  |


- 🏆 Modelo Selecionado: **XGBoost**

O XGBoost demonstrou desempenho superior em **todas as métricas relevantes**, sendo altamente eficaz para **prever a saída de funcionários**.

| Métrica           | Valor  | Interpretação Prática                                                  |
|-------------------|--------|-----------------------------------------------------------------------|
| **Recall (1)**       | 0.9718 | Detecta 97% dos funcionários que realmente sairão.                   |
| **Precision (1)**    | 1.0000 | Sempre que prevê rotatividade, está correto (zero falsos positivos).  |
| **Acurácia**         | 0.9955 | Classifica corretamente quase todos os casos.                        |
| **F1-Score**         | 0.9857 | Excelente equilíbrio entre precisão e recall.                        |
| **ROC-AUC**          | 0.9859 | Excelente separação entre as classes.                                |


🌟 **Variáveis Mais Relevantes (Análise pelo Random Forest)**

A análise de importância das variáveis revela os **principais fatores de risco** para rotatividade:

| Posição | Variável               | Importância Relativa | Insight de Negócio                                                             |
|---------|------------------------|---------------------|-------------------------------------------------------------------------------|
| 1º      | `TotalWorkingYears`     | 0.0909              | Funcionários com mais experiência têm maior risco de sair.                    |
| 2º      | `Age`                   | 0.0806              | Faixas etárias específicas apresentam maior risco (ex: jovens e pré-aposentadoria). |
| 3º      | `YearsWithCurrManager`  | 0.0802              | Relação com o gestor direto influencia fortemente na permanência.             |
| 4º      | `YearsAtCompany`        | 0.0782              | Pode indicar estagnação na carreira (falta de promoções, desafios).            |
| 5º      | `MonthlyIncome`         | 0.0778              | Fator financeiro permanece decisivo na decisão de permanência.                 |

✅ **Conclusão**

- A modelagem supervisionada permitiu **prever com alta precisão** a rotatividade dos funcionários.
- O **XGBoost** foi o modelo campeão, com Recall de **97%** e zero falsos positivos.
- As variáveis de maior impacto possuem **interpretações práticas diretas para ações do RH**.
- Essa abordagem possibilita **intervenções proativas para retenção de talentos em risco**, aumentando a eficácia das políticas internas.
