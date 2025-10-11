## 🟦 2.1 Processar e preparar base de dados

📌 O objetivo principal desta etapa foi inspecionar, limpar, transformar e estruturar os dados para que estivessem prontos para o treinamento do modelo de machine learning.

### 2.1.1 🔵 Conectar/importar dados para outras ferramentas

Nesta etapa inicial, realizamos a conexão e importação dos dados para viabilizar a análise exploratória e o pré-processamento. Para isso, utilizamos a linguagem **Python** e a biblioteca **pandas**, amplamente empregada para manipulação e análise de dados.

O arquivo denominado `rh_data.csv` está localizado no diretório:

```
C:\Users\Thais Lira\Documents\rota-complementar-Machine-learning
```

O arquivo foi carregado com sucesso em um DataFrame chamado `df`, permitindo a manipulação e exploração das informações.

- 📊 A base contém **24 variáveis (colunas)** e **4.410 registros (linhas)**, cada um representando um funcionário distinto.

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

🔎 **Valores Nulos**

Foram identificados valores ausentes nas colunas NumCompaniesWorked (19) e TotalWorkingYears (9).

✅ Ação: Preenchimento com a mediana

💡 Justificativa: Valores extremos nas colunas poderiam distorcer a média. A mediana é mais robusta nesse caso.

🔎 **Valores Duplicados**

Nenhuma linha duplicada foi encontrada.

✅ Ação: Nenhuma necessária.

🔎 **Valores Fora do Escopo da Análise**

A variável `Over18` forai identificada como redundante.

✅ Ação: Remoção das variáveis Over18.

💡 Justificativa: A remoção dessas colunas de baixa ou nenhuma variância evita ruído no modelo.

Foi identificado desbalanceamento na variável-alvo Attrition:

| Classe | Quantidade | Percentual |
| :--- | :--- | :--- |
| No | 3.699 | 84% |
| Yes | 711 | 16% |

✅ Ação futura: Aplicar técnicas de balanceamento durante o treinamento do modelo (como SMOTE ou ajuste de pesos).

🔎 **Analise das variáveis Categoricas**

A base de dados possui 8 variáveis categóricas que representam atributos qualitativos dos funcionários, fundamentais para a análise preditiva da rotatividade (Attrition). Estas variáveis incluem dados sobre perfil, função, departamento, estado civil, entre outros.

**Consistência**: Foi realizada a inspeção dos valores únicos para garantir que não existam dados inválidos ou inconsistentes.

**Nulos**: Não foram encontrados valores ausentes nas variáveis categóricas.

**Balanceamento**: A variável alvo Attrition apresenta desbalanceamento (16% sim, 84% não), a ser tratado na fase de modelagem.

**Codificação para Modelagem**

Para que os algoritmos de machine learning possam processar essas variáveis, todas foram convertidas para formato numérico via Label Encoding, utilizando o LabelEncoder do sklearn.preprocessing.

- **Motivação para Label Encoding:**

 - Compatível com modelos baseados em árvores, como XGBoost, que conseguem interpretar códigos numéricos mesmo que representem categorias nominais.

 - Evita aumento no número de variáveis (como ocorreria no one-hot encoding), mantendo a dimensionalidade controlada.

- *Variáveis Codificadas*:

- Attrition, BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, AgeGroup, DistanceCategory

**Impacto da Codificação**

- A codificação viabiliza o uso do conjunto completo de dados no modelo XGBoost.

- Garante que as variáveis categóricas contribuam com sua informação sem causar erros na fase de treinamento.

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

✅ Ação: Durante a análise exploratória dos dados, foram identificados valores considerados outliers em algumas variáveis numéricas, tais como MonthlyIncome, TotalWorkingYears, TrainingTimesLastYear, entre outras.

No entanto, para o desenvolvimento do modelo preditivo, optamos por utilizar o algoritmo XGBoost, que é baseado em árvores de decisão e reconhecidamente robusto à presença de outliers. Diferentemente de modelos lineares ou baseados em distância, o XGBoost não é significativamente afetado por valores extremos, pois utiliza regras de divisão para particionar os dados.

Dessa forma, decidimos manter as variáveis numéricas em sua forma original, sem realizar qualquer tratamento específico para outliers, preservando a integridade dos dados e evitando possíveis distorções decorrentes de transformações ou remoção de pontos.

Essa abordagem visa garantir que o modelo possa capturar padrões importantes, incluindo informações relevantes que os valores extremos podem representar no contexto da rotatividade de funcionários.

🔎  **Verificar os tipos de dados**

Foi utilizado o comando `df.dtypes` para listar os tipos de dados.

- Variáveis Numéricas (16):

| Variável                  | Descrição                                   |
| ------------------------- | ------------------------------------------- |
| `Age`                     | Idade do funcionário                        |
| `DistanceFromHome`        | Distância da casa até o trabalho            |
| `Education`               | Nível de escolaridade (1 a 5)               |
| `EmployeeCount`           | Contagem de funcionários (valor fixo: 1)    |
| `EmployeeID`              | ID do funcionário                           |
| `JobLevel`                | Nível do cargo                              |
| `MonthlyIncome`           | Renda mensal                                |
| `NumCompaniesWorked`      | Nº de empresas em que já trabalhou          |
| `PercentSalaryHike`       | Aumento percentual de salário               |
| `StandardHours`           | Horas padrão (valor fixo: 8)                |
| `StockOptionLevel`        | Nível de participação em ações              |
| `TotalWorkingYears`       | Total de anos de experiência                |
| `TrainingTimesLastYear`   | Nº de treinamentos realizados no último ano |
| `YearsAtCompany`          | Anos na empresa atual                       |
| `YearsSinceLastPromotion` | Anos desde a última promoção                |
| `YearsWithCurrManager`    | Anos com o atual gerente                    |

- Variáveis Categóricas (8):

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

📌 O objetivo desta etapa foi dividir o conjunto de dados em duas partes: uma para o treinamento do modelo (80%) e outra para a sua validação (20%), garantindo reprodutibilidade e a mesma proporção da variável alvo em ambos os conjuntos.

✅ Técnicas utilizadas:

- **Função:** `train_test_split` do pacote `sklearn.model_selection`
- **Parâmetros definidos:**
  - `test_size=0.2`: 20% dos dados para teste
  - `random_state=42`: garante reprodutibilidade
  - `stratify=y`: mantém a proporção da variável-alvo (`Attrition`)

✅ Resultado:

| Conjunto     | Proporção "No" | Proporção "Yes" |
|--------------|----------------|-----------------|
| Geral        | ~84%           | ~16%            |
| Treinamento  | ~84%           | ~16%            |
| Teste        | ~83,9%         | ~16,1%          |

💡 Garantir a proporcionalidade é fundamental para evitar viés na avaliação do modelo e assegurar que ele generalize bem para novos dados.

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
