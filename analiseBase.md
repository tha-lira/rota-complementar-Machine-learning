# 📊 Previsão de Rotatividade de Funcionários com Machine Learning

## 1. Contexto

No mercado atual, reter talentos é uma estratégia essencial. A previsão da rotatividade (attrition) auxilia a empresa a atuar preventivamente, reduzindo custos com recrutamento e aumentando a eficiência na gestão de pessoas.

Nosso objetivo é construir um modelo supervisionado de machine learning para prever se um funcionário sairá ou não da empresa (classificação binária).

## 🟦 2.1 Processar e preparar base de dados

📌 O objetivo principal desta etapa foi importar, inspecionar e preparar os dados fornecidos pelo departamento de Recursos Humanos, garantindo que estejam prontos para o treinamento do modelo supervisionado de machine learning. Este pré-processamento é fundamental para assegurar a qualidade, consistência e estrutura adequada dos dados que serão utilizados.

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

2.1.2 🔵Limpar dados

📌 A limpeza de dados é uma etapa fundamental antes do treinamento de qualquer modelo de machine learning. Seu principal objetivo é garantir a qualidade, consistência e confiabilidade da base, minimizando ruídos e distorções que possam impactar a performance do modelo preditivo.

🔎 **Valores Nulos**

Foram identificados valores ausentes nas seguintes colunas:

| Coluna             | Valores Nulos |
| ------------------ | ------------- |
| NumCompaniesWorked | 19            |
| TotalWorkingYears  | 9             |

✅ Ação tomada: Preenchimento com a **mediana**  
💡 Justificativa: Valores extremos nas colunas poderiam distorcer a média. A mediana é mais robusta nesse caso.

🔎 **Valores Duplicados**

- Nenhuma linha duplicada foi encontrada.

✅ Ação: Nenhuma necessária.

🔎 **Valores Fora do Escopo da Análise**

- A variável `Over18` foi identificada como redundante (possui apenas o valor "Y").

✅ Ação: Remoção da variável `Over18`.  
💡 Justificativa: Já temos a variável `Age`, que fornece essa informação de forma mais útil.

🔎 **Valores Discrepantes em Variáveis Categóricas**

- Foi identificado **desbalanceamento** na variável-alvo `Attrition`:

| Classe | Quantidade | Percentual |
|--------|------------|------------|
| No     | 3.699      | 84%        |
| Yes    | 711        | 16%        |

✅ Ação futura: Aplicar técnicas de balanceamento durante o treinamento do modelo (como SMOTE ou ajuste de pesos).

🔎 **Valores Discrepantes (outliers) em variáveis Numéricas**

Utilizou-se o **IQR (Intervalo Interquartil)** para identificar possíveis outliers em variáveis numéricas.

##### Variáveis sem outliers:

- `Age`
- `DistanceFromHome`
- `Education`
- `EmployeeCount`
- `EmployeeID`
- `JobLevel`
- `PercentSalaryHike`
- `StandardHours`

##### Variáveis com maior incidência de outliers:

| Variável                  | Outliers Identificados | Observações                               |
| ------------------------- | ---------------------- | ------------------------------------------ |
| `MonthlyIncome`           | 342                    | Valores muito altos                        |
| `NumCompaniesWorked`      | 156                    | Muitos registros no limite (9)             |
| `StockOptionLevel`        | 255                    | Concentração no valor máximo               |
| `TotalWorkingYears`       | 189                    | Experiência elevada                        |
| `TrainingTimesLastYear`   | 714                    | Concentração no valor máximo (6)           |
| `YearsAtCompany`          | 312                    | Registros com até 40 anos na empresa       |
| `YearsSinceLastPromotion` | 321                    | Funcionários sem promoção por 15 anos      |
| `YearsWithCurrManager`    | 42                     | Até 17 anos com o mesmo gerente            |

✅ Ação: Ainda em análise. Pode-se considerar técnicas como normalização ou remoção seletiva.

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

### 🔎 Codificar variáveis categóricas

📌 Para treinar modelos de machine learning, todas as variáveis devem estar no formato numérico.

✅ Ação: Aplicado **Label Encoding** com `LabelEncoder` da biblioteca `sklearn.preprocessing`.

#### Variáveis transformadas:

- `Attrition`
- `BusinessTravel`
- `Department`
- `EducationField`
- `Gender`
- `JobRole`
- `MaritalStatus`

💡 Justificativa: Label Encoding é apropriado para variáveis ordinais ou nominais em modelos baseados em árvores (como Decision Trees e Random Forests).

2.1.3 🔵 Dividir a base em treino e teste

📌 **Objetivo:**  
Dividir o conjunto de dados em dois subconjuntos: um para **treinamento** e outro para **teste**, garantindo reprodutibilidade e representatividade da variável-alvo (Attrition).

---

💡 **Importância da Divisão Treino/Teste**

2.1.3 🔵 Dividir a base em treino e teste

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

2.1.4 🔵 Criar novas variáveis

### 2.1.4 🔵 Criar novas variáveis (Feature Engineering)

**Objetivo:**  
Criar novas variáveis explicativas a partir das características existentes na base de dados, com o intuito de melhorar a capacidade preditiva do modelo.

---

**Descrição das ações realizadas:**

- **Faixa Etária (`AgeGroup`):**  
  Criamos uma nova variável categórica que agrupa a idade dos funcionários em três categorias:  
  - *Jovem*: até 30 anos  
  - *Adulto*: entre 31 e 45 anos  
  - *Sênior*: acima de 45 anos  
  Essa variável permite capturar efeitos relacionados a diferentes fases da carreira e idade, que podem influenciar a rotatividade.

- **Categoria de Distância (`DistanceCategory`):**  
  A variável numérica `DistanceFromHome` foi convertida em categorias baseadas em intervalos:  
  - *Perto*: até 5 km  
  - *Médio*: entre 6 e 15 km  
  - *Longe*: acima de 15 km  
  Essa categorização facilita a identificação de padrões relacionados à distância da residência ao trabalho.

- **Indicador de Muitos Empregos Anteriores (`ManyCompaniesWorked`):**  
  Criamos uma variável binária que indica se o funcionário trabalhou em mais de 3 empresas anteriormente. Essa informação pode ser relevante para identificar perfis de maior mobilidade profissional.

---

**Observações importantes:**  
- As variáveis categóricas criadas (`AgeGroup` e `DistanceCategory`) inicialmente possuem valores em formato texto, sendo necessário codificá-las posteriormente para uso em modelos de machine learning (por exemplo, via Label Encoding ou One-Hot Encoding).  
- A variável binária `ManyCompaniesWorked` já está em formato numérico, pronta para ser utilizada diretamente.





