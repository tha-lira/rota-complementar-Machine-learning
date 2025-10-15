## 🟦 2.1 Processar e preparar base de dados

📌 O objetivo principal desta etapa foi inspecionar, limpar, transformar e estruturar o conjunto de dados para que estivesse pronto para o treinamento do modelo de Machine Learning.

### 2.1.1 🔵 Conectar/importar dados para outras ferramentas

Nesta etapa inicial, o arquivo `rh_data.csv` foi importado utilizando a biblioteca `pandas`.

📊 A base original continha 26 variáveis (colunas) e 4.410 registros (linhas), cada um representando um funcionário.

A seguir, apresentamos a descrição das variáveis que compõem a tabela:

| Variável                    | Descrição                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Age**                     | Idade do funcionário                                                                                         |
| **Attrition**               | Funcionário que saiu da empresa (0 = não, 1 = sim)                                                           |
| **BusinessTravel**          | Frequência com que o funcionário viaja                                                                       |
| **Department**              | Departamento em que o funcionário trabalha                                                                   |
| **DistanceFromHome**        | Distância da casa do funcionário até a empresa                                                               |
| **Education**               | Nível de escolaridade (1 = Abaixo da faculdade, 2 = Universidade, 3 = Bacharelado, 4 = Mestrado, 5 = Doutor) |
| **EducationField**          | Área de estudo dos funcionários                                                                              |
| **EmployeeCount**           | Contagem de funcionários                                                                                     |
| **EmployeeID**              | Número de identificação do funcionário                                                                       |
| **Gender**                  | Sexo do funcionário                                                                                          |
| **JobLevel**                | Nível da função do funcionário na empresa                                                                    |
| **JobRole**                 | Nome da função do funcionário                                                                                |
| **MaritalStatus**           | Estado civil do funcionário                                                                                  |
| **MonthlyIncome**           | Renda mensal do funcionário                                                                                  |
| **NumCompaniesWorked**      | Número de empresas em que o funcionário já trabalhou                                                         |
| **Over18**                  | Indica se o funcionário tem mais de 18 anos (verdadeiro/falso)                                               |
| **PercentSalaryHike**       | Aumento percentual do salário                                                                                |
| **StandardHours**           | Horário de trabalho padrão                                                                                   |
| **StockOptionLevel**        | Participação em ações (quanto maior o número, mais opções de ações um funcionário tem)                       |
| **TotalWorkingYears**       | Anos trabalhados                                                                                             |
| **TrainingTimesLastYear**   | Total de horas dedicadas ao treinamento no último ano                                                        |
| **YearsAtCompany**          | Anos trabalhados nesta empresa                                                                               |
| **YearsSinceLastPromotion** | Anos desde a última promoção                                                                                 |
| **YearsWithCurrManager**    | Anos trabalhando com o gerente atual                                                                         |

### 2.1.2 🔵 Limpar dados

📌 A limpeza de dados é uma etapa essencial na preparação de um modelo de machine learning. Seu objetivo é assegurar a qualidade e a consistência das informações, reduzindo ruídos e distorções que possam comprometer a precisão preditiva do modelo.

✅ Variável-Alvo

| Variável    | Descrição                              |
| ----------- | -------------------------------------- |
| `Attrition` | Variável-alvo (0 = não saiu, 1 = saiu) |

🔎 **Valores Nulos**

Foram identificados valores ausentes nas colunas:

| Coluna             | Qtde. Nulos |
| ------------------ | ----------- |
| NumCompaniesWorked | 19          |
| TotalWorkingYears  | 9           |

✅ Ação: preenchimento dos valores nulos com a **mediana** de cada coluna, uma vez que a mediana é robusta contra valores extremos e representa melhor a tendência central dos dados neste contexto.

🔎 **Valores Duplicados**

Nenhuma linha duplicada foi encontrada.

✅ Ação: Nenhuma necessária.

🔎 **Valores Fora do Escopo da Análise**

Quatro colunas foram removidas por não agregarem valor preditivo ou por serem constantes:

| Variável        | Justificativa                                  |
| --------------- | ---------------------------------------------- |
| `EmployeeCount` | Constante (valor fixo)                         |
| `StandardHours` | Constante (valor fixo)                         |
| `EmployeeID`    | Identificador único, sem valor preditivo       |
| `Over18`        | Constante (todos os registros indicavam 'Sim') |

✅ Ação: Remoção da coluna `Over18`, `EmployeeID`, `StandardHours`, `EmployeeCount` para evitar ruídos e redundância no modelo.

🔎 **Análise do Desbalanceamento da Variável-Alvo (Attrition)**

A variável Attrition está significativamente desbalanceada, o que impacta a escolha de métricas e abordagens.

| Classe  | Quantidade | Percentual |
| ------- | ---------- | ---------- |
| Não (0) | 3.699      | 83.88%     |
| Sim (1) | 711        | 16.12%     |

✅ Conclusão: A rotatividade é uma classe minoritária. Portanto, métricas como F1-Score e Recall devem ser priorizadas, e técnicas de balanceamento (pesos ou sampling) podem ser necessárias na fase de modelagem.

🔎 **Analise das variáveis Categoricas**

A base contém 8 variáveis categóricas essenciais para a análise da rotatividade. Todas passaram por verificação de consistência e ausência de valores nulos.

Todas as variáveis categóricas foram convertidas para formato numérico utilizando `Label Encoding` (LabelEncoder do scikit-learn).

🔎 **Analise das variáveis Numéricas (outliers)**

- Métodos utilizados:

 - Detecção por desvio padrão (±3σ)
 - Detecção pelo método do Intervalo Interquartil (IQR)

Resumo dos resultados:

| Variável                | Média    | Desvio Padrão | Limite Inferior (IQR) | Limite Superior (IQR) | Qtde de Outliers (IQR) | Observações                                                              |
| ----------------------- | -------- | ------------- | --------------------- | --------------------- | ---------------------- | ------------------------------------------------------------------------ |
| MonthlyIncome           | 65029.31 | 47068.89      | -52925.00             | 165835.00             | 342                    | Presença significativa de outliers em salários muito altos               |
| NumCompaniesWorked      | 2.69     | 2.50          | -3.50                 | 8.50                  | 156                    | Alguns funcionários com histórico de trabalho extenso em várias empresas |
| StockOptionLevel        | 0.79     | 0.85          | -1.50                 | 2.50                  | 255                    | Possibilidade de valores máximos em opções de ações                      |
| TotalWorkingYears       | 11.28    | 7.78          | -7.50                 | 28.50                 | 189                    | Outliers indicam funcionários com experiência muito elevada              |
| TrainingTimesLastYear   | 2.80     | 1.29          | 0.50                  | 4.50                  | 714                    | Grande variação na quantidade de treinamentos realizados                 |
| YearsAtCompany          | 7.01     | 6.13          | -6.00                 | 18.00                 | 312                    | Funcionários com longo tempo de casa                                     |
| YearsSinceLastPromotion | 2.19     | 3.22          | -4.50                 | 7.50                  | 321                    | Alguns funcionários sem promoções há bastante tempo                      |
| YearsWithCurrManager    | 4.12     | 3.57          | -5.50                 | 14.50                 | 42                     | Variabilidade na relação com o gestor atual                              |

✅ Ação: Durante a análise exploratória dos dados, foram identificados valores considerados outliers em algumas variáveis numéricas, como MonthlyIncome, TotalWorkingYears e TrainingTimesLastYear.

Optamos por não removê-los, pois o modelo escolhido — `Random Forest` — é baseado em árvores de decisão e, portanto, **robusto à presença de outliers**.
Diferentemente de modelos lineares, ele utiliza divisões baseadas em regras, o que reduz o impacto de valores extremos no desempenho.

Ainda assim, reconhecemos que a presença de outliers pode contribuir para um leve sobreajuste (overfitting), dado o alto desempenho observado (acurácia de 97%). Em futuras iterações, seria interessante realizar testes complementares com validação cruzada e modelos regularizados, garantindo a generalização dos resultados.

Apesar do excelente desempenho, foi mantida uma postura crítica quanto à possibilidade de overfitting, já que resultados muito altos em bases pequenas podem refletir uma adaptação excessiva do modelo ao conjunto de treino.

🔎  **Verificar os tipos de dados**

Foi utilizado o comando `df.dtypes` para listar os tipos de dados.

- Variáveis Numéricas (13):

| Variável                  | Descrição                                   |
| ------------------------- | ------------------------------------------- |
| `Age`                     | Idade do funcionário                        |
| `DistanceFromHome`        | Distância da casa até o trabalho            |
| `Education`               | Nível de escolaridade (1 a 5)               |
| `JobLevel`                | Nível do cargo                              |
| `MonthlyIncome`           | Renda mensal                                |
| `NumCompaniesWorked`      | Nº de empresas em que já trabalhou          |
| `PercentSalaryHike`       | Aumento percentual de salário               |
| `StockOptionLevel`        | Nível de participação em ações              |
| `TotalWorkingYears`       | Total de anos de experiência                |
| `TrainingTimesLastYear`   | Nº de treinamentos realizados no último ano |
| `YearsAtCompany`          | Anos na empresa atual                       |
| `YearsSinceLastPromotion` | Anos desde a última promoção                |
| `YearsWithCurrManager`    | Anos com o atual gerente                    |

- Variáveis Categóricas (7):

| Variável         | Descrição                                 |
| ---------------- | ------------------------------------------ |
| `Attrition`      | Se o funcionário saiu da empresa (Yes/No) |
| `BusinessTravel` | Frequência de viagens a trabalho          |
| `Department`     | Departamento onde trabalha                |
| `EducationField` | Área de formação acadêmica                |
| `Gender`         | Gênero                                    |
| `JobRole`        | Cargo ocupado                             |
| `MaritalStatus`  | Estado civil                              |

### 2.1.3 🔵 Dividir a base em treino e teste

Essa etapa garante avaliação do modelo em dados nunca vistos, simulando seu desempenho em produção.

✅ Técnicas Utilizadas

- Divisão: `train_test_split` (80% Treino, 20% Teste)

- Reprodutibilidade: `random_state=42`

- Estratificação: `stratify=y`, mantendo a proporção da variável alvo.

✅ Resultado da Divisão

| Conjunto    | Proporção "Não" | Proporção "Sim" |
| ----------- | --------------- | --------------- |
| Treinamento | 83.87%          | 16.13%          |
| Teste       | 83.90%          | 16.10%          |

💡 Conclusão: A estratificação foi bem-sucedida, garantindo que o conjunto de teste seja representativo do conjunto geral.

### 2.1.4 🔵 Criar novas variáveis (Feature Engineering)

📌 Objetivo: Criar novas variáveis explicativas a partir das características existentes na base de dados, com o intuito de melhorar a capacidade preditiva do modelo.

**Descrição das ações realizadas:**

- **Faixa Etária (`AgeGroup`):**  
  - *Jovem*: até 30 anos  
  - *Adulto*: entre 31 e 45 anos  
  - *Sênior*: acima de 45 anos  

- **Categoria de Distância (`DistanceCategory`):**  
  - *Perto*: até 5 km  
  - *Médio*: entre 6 e 15 km  
  - *Longe*: acima de 15 km  

- **Indicador de Muitos Empregos Anteriores (`ManyCompaniesWorked`):**  
 - Variável binária (0 ou 1) que indica se o funcionário trabalhou em mais de 3 empresas.

**Observações importantes:**  
Todas as variáveis criadas (incluindo AgeGroup e DistanceCategory) foram subsequentemente codificadas para o formato numérico (Label Encoding), tornando-as prontas para o treinamento do modelo.
