🟪 2.2 Fazer uma análise exploratória

Esta etapa consistiu em uma análise descritiva e visual dos dados, essencial para compreender o comportamento das variáveis e identificar padrões associados à variável-alvo, a rotatividade (Attrition).

🔍 O desbalanceamento da variável-alvo foi confirmado: apenas 16,12% dos funcionários deixaram a empresa, reforçando o foco em métricas da classe minoritária nas fases seguintes do projeto.

### 🟣 2.2.1 Agrupar dados de acordo com variáveis ​​categóricas

A análise categórica agrupada pela taxa média de rotatividade (Attrition = 1) revela perfis de maior risco, com forte potencial de aplicação estratégica.

### 🔹 Taxa de Rotatividade por Departamento

| Departamento              | Total | Desligados | Taxa (%) |
| ------------------------- | ----- | ---------- | -------- |
| Human Resources (0)       | 189   | 57         | 30,16%   |
| Research & Development(1) | 2883  | 453        | 15,71%   |
| Sales (2)                 | 1338  | 201        | 15,02%   |

📌 **Insight Crítico:** O departamento de RH tem a maior taxa de desligamento, sugerindo possíveis desafios internos neste setor.

### 🔹 Taxa de Rotatividade por Faixa Etária (AgeGroup)

| Faixa Etária | Total | Desligados | Taxa (%) |
| ------------ | ----- | ---------- | -------- |
| Jovem (0)    | 2433  | 309        | 12,71%   |
| Adulto (1)   | 1158  | 300        | 25,91%   |
| Sênior (2)   | 819   | 102        | 12,45%   |

🔍 **Insight:** Profissionais adultos (31–45 anos) apresentam maior rotatividade em comparação com jovens e seniores.

🔹 Rotatividade por Distância de Casa (DistanceCategory)

| Distância | Total | Desligados | Taxa (%) |
| --------- | ----- | ---------- | -------- |
| Perto (0) | 987   | 147        | 14,89%   |
| Médio (1) | 1527  | 261        | 17,10%   |
| Longe (2) | 1896  | 303        | 15,97%   |

🚗 Insight: Funcionários que moram em distância média da empresa apresentam ligeiramente maior probabilidade de desligamento.

🔹 Outros Fatores de Risco

| Variável            | Categoria                    | Taxa de Rotatividade |
| ------------------- | ---------------------------- | -------------------- |
| MaritalStatus       | Solteiro(a)                  | 25,5% (estimado)     |
| Gender              | Masculino                    | 16,7% (estimado)     |
| ManyCompaniesWorked | Trabalhou em muitas empresas | Elevada              |

💡 Destaque: Ter trabalhado em muitas empresas anteriormente está associado a maior rotatividade, indicando possível instabilidade.

### 🟣 2.2.2 Visualizar as variáveis categóricas

Gráficos de barras foram gerados para ilustrar as taxas médias de `Attrition` por categoria de cada variável. As imagens estão salvas na pasta `/figuras` do repositório.

### 📊 Distribuição de Idade e Renda
![Distribuição](figuras/distribuicao_idade_renda.png)

### 🚻 Rotatividade por Gênero
![Gênero](figuras/rotatividade_gender.png)

### 💍 Rotatividade por Estado Civil
![Estado Civil](figuras/rotatividade_maritalstatus.png)

### 💼 Rotatividade por Cargo
![Cargo](figuras/rotatividade_jobrole.png)

### 🏢 Rotatividade por Departamento
![Departamento](figuras/rotatividade_department.png)

### 📈 Rotatividade por Faixa Etária
![Faixa Etária](figuras/rotatividade_agegroup.png)

### 🟣 2.2.3 Aplicar medidas de tendência central

Foi calculada a **média** e a **mediana** das principais variáveis numéricas para entender os valores típicos do conjunto de dados:

| Variável          | Média      | Mediana   |
| ----------------- | ---------- | --------- |
| Idade (Age)       | 36,92 anos | 36 anos   |
| Renda Mensal      | R$ 65.029  | R$ 49.190 |
| TotalWorkingYears | 11,28 anos | 10 anos   |
| YearsAtCompany    | 7,01 anos  | 5 anos    |

🧮 A renda mensal apresenta distribuição assimétrica à direita, com média superior à mediana, indicando presença de altos salários fora da curva.

### 🟣 2.2.4 Visualizar distribuição

A forma das distribuições foi analisada com base em histogramas:

- **Age (Idade):** distribuição razoavelmente simétrica, com concentração entre 30 e 45 anos.  
- **MonthlyIncome (Renda Mensal):** claramente assimétrica à direita, reforçando o impacto de salários altos sobre a média.

### 🟣 2.2.5 Aplicar medidas de dispersão 

As **medidas de dispersão** complementam a análise descritiva, fornecendo uma ideia da variação dos dados:

| Variável           | Desvio Padrão | Mínimo   | Máximo    |
| ------------------ | ------------- | -------- | --------- |
| Idade              | 9,13          | 18       | 60        |
| Renda Mensal       | R$ 47.068     | R$10.090 | R$199.990 |
| TotalWorkingYears  | 7,77          | 0        | 40        |
| NumCompaniesWorked | 2,49          | 0        | 9         |


> ⚠️ **Observação:** A alta dispersão salarial evidencia **grande desigualdade de renda**, o que pode ser um fator de insatisfação e contribuir para a rotatividade.


💡 Insights Finais da EDA

1. 🔴 Sinais de Alerta no RH

- O setor de Recursos Humanos tem a maior taxa de rotatividade.

- Adultos (31–45 anos) também apresentam risco elevado — provavelmente por serem mais experientes e exigirem mais reconhecimento.

2. 💸 Renda como Fator de Disparidade

- A alta dispersão de salários pode gerar insatisfação interna e sensação de injustiça.

- Funciona como proxy para políticas de valorização (ou sua ausência).

3. 🔁 Estabilidade Comportamental como Previsor

- Funcionários solteiros e com histórico de múltiplos empregos têm maior tendência a sair.

- Essas variáveis comportamentais e sociais devem ser consideradas em planos de retenção e engajamento.

🔎 Conclusão: Esta análise exploratória fornece base robusta para a etapa de modelagem preditiva, justificando a seleção e transformação de variáveis, e orientando o foco em perfis de maior risco de saída.