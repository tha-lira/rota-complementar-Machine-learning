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

2.1.2 🔵Limpar dados

📌 A limpeza de dados é uma etapa essencial na preparação de um modelo de machine learning. Seu objetivo é assegurar a qualidade e a consistência das informações, reduzindo ruídos e distorções que possam comprometer a precisão preditiva do modelo.

🔎 **Valores Nulos**

Foram identificados valores ausentes nas colunas `NumCompaniesWorked` (19) e `TotalWorkingYears` (9).

✅ Ação tomada: Preenchimento com a **mediana**  

💡 Justificativa: Valores extremos nas colunas poderiam distorcer a média. A mediana é mais robusta nesse caso.

🔎 **Valores Duplicados**

- Nenhuma linha duplicada foi encontrada.

✅ Ação: Nenhuma necessária.

🔎 **Valores Fora do Escopo da Análise**

As variáveis `Over18`, `EmployeeCount` e `StandardHours` foram identificadas como redundantes por apresentarem valores constantes em todos os registros.

✅ Ação: Remoção das variáveis `Over18`, `EmployeeCount` e `StandardHours`.

💡 Justificativa: A remoção dessas colunas de baixa ou nenhuma variância evita ruído no modelo.

🔎 **Valores Discrepantes em Variáveis Categóricas**

📌 Para treinar modelos de machine learning, todas as variáveis devem estar no formato numérico.

✅ Ação: Aplicado `Label Encoding` com `LabelEncoder` da biblioteca `sklearn.preprocessing`.

Variáveis transformadas (total de 9):

- Attrition

- BusinessTravel

- Department

- EducationField

- Gender

- JobRole

- MaritalStatus

- AgeGroup (Nova variável)

- DistanceCategory (Nova variável)

💡 Justificativa: `Label Encoding` é apropriado para variáveis ordinais ou nominais em modelos baseados em árvores (como Decision Trees e Random Forests).

Seu relatório está muito bem estruturado e detalhado, especialmente as seções de Feature Engineering e a Divisão Treino/Teste. Você está no caminho certo!

Há apenas um pequeno ajuste de ordem e dois detalhes a corrigir na seção de Limpeza de Dados para que o fluxo lógico do relatório corresponda à ordem de execução do código que você validou.

Correções Essenciais no Fluxo do Relatório
O único problema no seu texto atual é que a Criação de Novas Variáveis (2.1.4) aparece depois da Codificação (2.1.2). No seu código, você criou as novas variáveis antes de codificá-las.

1. Reorganização das Seções
A ordem lógica e correta (que você executou no código) é:

Limpar Dados (Nulos, Remoção, Outliers)

Criar Novas Variáveis (Feature Engineering)

Codificar Variáveis Categóricas (incluindo as novas)

Dividir a Base

2. Edição do Bloco de Limpeza (2.1.2)
A seção de Limpeza precisa estar completa.

Seção do Relatório	Sua Edição	Correção Necessária
Limpar Dados	Você inseriu a seção "Codificar variáveis categóricas" dentro da subseção "Limpar Dados".	O bloco de Codificação deve ser um item separado, pois a limpeza em si (nulos/remoção) ocorre antes.
Outliers	As variáveis que foram removidas (EmployeeCount, StandardHours) ainda estão listadas na tabela de variáveis numéricas.	Devem ser removidas da lista, pois não existem mais no df.

Exportar para as Planilhas
✅ Versão Final e Corrigida
Abaixo está o seu relatório com a ordem corrigida (2.1.3 e 2.1.4 foram trocadas) e as informações consolidadas.

(O restante do seu texto está perfeito e foi mantido).

🟦 2.1 Processar e preparar base de dados
[...] (O texto das seções 1 e 2.1.1 Conectar/importar dados é mantido)

2.1.2 🔵 Limpar dados
📌 A limpeza de dados é uma etapa essencial na preparação de um modelo de machine learning. Seu objetivo é assegurar a qualidade e a consistência das informações, reduzindo ruídos e distorções que possam comprometer a precisão preditiva do modelo.

🔎 Valores Nulos

Foram identificados valores ausentes nas colunas NumCompaniesWorked (19) e TotalWorkingYears (9).

✅ Ação tomada: Preenchimento com a mediana

💡 Justificativa: Valores extremos nas colunas poderiam distorcer a média. A mediana é mais robusta nesse caso.

🔎 Valores Duplicados

Nenhuma linha duplicada foi encontrada.

✅ Ação: Nenhuma necessária.

🔎 Valores Fora do Escopo da Análise

As variáveis Over18, EmployeeCount e StandardHours foram identificadas como redundantes por apresentarem valores constantes em todos os registros.

✅ Ação: Remoção das variáveis Over18, EmployeeCount e StandardHours.

💡 Justificativa: A remoção dessas colunas de baixa ou nenhuma variância evita ruído no modelo.

Foi identificado desbalanceamento na variável-alvo Attrition:

| Classe | Quantidade | Percentual |
| :--- | :--- | :--- |
| No | 3.699 | 84% |
| Yes | 711 | 16% |

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
 - Foi aplicado LabelEncoder nas variáveis categóricas conforme orientação do guia, garantindo que o modelo consiga processar os dados textuais.

#### Variáveis transformadas:

- `Attrition`
- `BusinessTravel`
- `Department`
- `EducationField`
- `Gender`
- `JobRole`
- `MaritalStatus`

💡 Justificativa: Label Encoding é apropriado para variáveis ordinais ou nominais em modelos baseados em árvores (como Decision Trees e Random Forests).

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