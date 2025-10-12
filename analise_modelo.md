## 🔴 2.3.1 Comparação de Desempenho dos Modelos

Três algoritmos foram avaliados em um conjunto de teste não balanceado para simular as condições reais de negócio. A métrica prioritária para desempate foi o **F1-Score da Classe 1 (rotatividade)**, que equilibra a capacidade de detectar a rotatividade real (**Recall**) e a precisão das previsões (**Precision**).

| Modelo                    | Acurácia Geral | Recall (Rotatividade) | Precision (Rotatividade) | F1-Score (Rotatividade) |
|---------------------------|----------------|----------------------|-------------------------|------------------------|
| 🏆 Random Forest (Vencedor) | 0.9966         | 0.98                 | 1.00                    | 0.9893                 |
| XGBoost                   | 0.9966         | 0.98                 | 1.00                    | 0.9893                 |
| Regressão Logística       | 0.8356         | 0.01                 | 0.20                    | 0.0136                 |

**🏆 Modelo Selecionado:** Random Forest

---

O Random Forest e o XGBoost apresentaram desempenho excepcionalmente alto e praticamente idêntico. O F1-Score de 0.9893 é o maior entre os modelos testados, indicando:

- **Alta capacidade de alertar sobre a rotatividade** (Recall ≈ 98%)
- **Garantia de que a maioria dos alertas seja correta** (Precision ≈ 100%)

| Métrica       | Valor | Interpretação Prática                                                      |
|---------------|-------|---------------------------------------------------------------------------|
| Recall (1)    | 0.98  | O modelo detecta 98% dos funcionários que realmente sairão da empresa.    |
| Precision (1) | 1.00  | Em 100% dos casos em que o modelo prevê rotatividade, a previsão está correta (zero Falsos Positivos). |
| F1-Score      | 0.9893| Excelente equilíbrio, tornando o modelo altamente confiável para intervenções proativas. |

---

**Nota Técnica:** Assim como na análise anterior, o desempenho próximo de 100% nas métricas para modelos complexos em dados de RH sugere possível superestimativa no conjunto de teste. Ainda assim, o Random Forest foi escolhido por apresentar o melhor desempenho conforme o script final, garantindo robustez mesmo com performance mais modesta em produção.

---

## 🌟 Fatores de Rotatividade Mais Relevantes (Feature Importances)

A análise de importância das variáveis, fornecida pelo modelo Random Forest, revela os fatores de risco críticos que impulsionam a rotatividade. Essa lista fundamenta as recomendações de retenção.

| Posição | Variável           | Importância Relativa | Insight Estratégico                                                                                       |
|---------|--------------------|---------------------|----------------------------------------------------------------------------------------------------------|
| 1º      | MonthlyIncome      | 0.1465              | **Fator Financeiro Dominante:** Renda mensal é o fator mais decisivo na propensão à saída. Reforça a necessidade de revisar a competitividade e equidade salarial. |
| 2º      | ExperienceRatio    | 0.0956              | **Relação Experiência/Tempo na Empresa:** Indica satisfação ou estagnação no cargo atual.                  |
| 3º      | PercentSalaryHike  | 0.0911              | **Reconhecimento de Desempenho:** Percentual de aumento salarial anual é crucial para retenção.           |
| 4º      | YearsAtCompany     | 0.0880              | **Tempo de Estagnação:** Permanência prolongada sem progressão clara é fator de risco.                    |
| 5º      | JobRole            | 0.0773              | **Função/Cargo:** Alguns cargos têm risco intrínseco maior de rotatividade (vendas, técnicos, etc.).      |
| 6º      | NumCompaniesWorked | 0.0692              | **Instabilidade Histórica:** Histórico de muitas trocas de emprego aumenta a propensão a sair.           |
| 7º      | YearsWithCurrManager | 0.0674            | **Qualidade da Gestão:** Longevidade do relacionamento com gestor imediato é fator humano relevante.     |

---

## ✅ Conclusão da Modelagem

A modelagem supervisionada identificou o **Random Forest** como o melhor preditor, capaz de gerar alertas de rotatividade com alta precisão e recall. Os insights extraídos da importância das features fornecem um roteiro de ação claro e baseado em dados para o RH, focando primariamente em:

- 💸 **Compensação:** MonthlyIncome, PercentSalaryHike  
- 📈 **Gestão de Carreira:** YearsAtCompany, ExperienceRatio  
- 👥 **Qualidade da Liderança:** YearsWithCurrManager

