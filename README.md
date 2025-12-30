# Previsão de custos de seguro saúde

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Concluído-green)

## Descrição do projeto
Este projeto de Ciência de Dados tem como objetivo analisar como diferentes atributos (idade, gênero, IMC, tabagismo) impactam os custos médicos individuais e construir um modelo preditivo capaz de estimar despesas futuras.

O projeto segue um fluxo completo: limpeza de dados, Análise Exploratória (EDA), engenharia de atributos e modelagem com Machine Learning.

## Principais insights de negócio
Através da análise exploratória, identificamos:

1.  **O fator crítico:** fumantes custam, em média, **4 vezes mais** que não fumantes.
2.  **A "Zona de perigo":** a obesidade (IMC >= 30) isoladamente tem um impacto moderado nos custos. Porém, a combinação **Fumante + Obeso** cria um efeito multiplicador, gerando os custos mais altos da carteira (acima de $40.000).
3.  **Idade:** existe uma progressão linear natural de custo com a idade, mas ela é secundária se comparada ao tabagismo.
4.  **Região:** a região *Southeast* apresenta os maiores custos médios e também a maior taxa de fumantes.

## 🛠️ Tecnologias utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, Numpy
* **Visualização:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (Pipeline, OneHotEncoder, StandardScaler, RandomForest)

## 🤖 Modelagem e performance
Foram testados dois algoritmos para a regressão. O modelo **Random Forest** apresentou desempenho superior por capturar as não-linearidades dos dados (especialmente a interação Fumo/Obesidade).

| Modelo | R² Score | RMSE (Erro Médio) | Observação |
| :--- | :--- | :--- | :--- |
| Regressão Linear | 0.78 | ~$6,000 | Baseline simples |
| **Random Forest** | **0.86** | **~$4,500** | **Melhor performance** |

> **Conclusão:** o modelo Random Forest explica 86% da variância dos custos, com um erro médio de aproximadamente $4,500.

## 📁 Estrutura do projeto
```text
├── data/              # Dataset original (insurance.csv)
├── notebooks/         # Jupyter Notebook com a análise completa
├── images/            # Gráficos gerados durante a análise
├── requirements.txt   # Bibliotecas necessárias
└── README.md          # Este arquivo
