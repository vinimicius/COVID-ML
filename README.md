# COVID-ML: Predição de Óbito por COVID-19 em SP

  

Este projeto aplica técnicas de **Aprendizado de Máquina Supervisionado** para prever o desfecho (óbito ou sobrevivência) de pacientes acometidos pela COVID-19 no Estado de São Paulo, correlacionando fatores como idade, gênero e comorbidades pré-existentes.

  

## 📋 Resumo do Projeto

O trabalho busca aplicar técnicas de aprendizado de máquina supervisionado para prever o óbito de pacientes acometidos pela COVID-19 de diferentes idades e com doenças pré-existentes. Diante da situação provocada pela pandemia entre 2020 e 2023, o objetivo é correlacionar a fatalidade a partir de dados de saúde e geográficos, analisando a relação entre o desfecho da doença e a presença de comorbidades.

  

## 📊 Conjunto de Dados

Os dados foram obtidos do repositório do **Sistema Estadual de Análise de Dados (SEADE)** de São Paulo.

* **Fonte:** [Repositório SEADE - COVID-19](https://repositorio.seade.gov.br/group/covid-19)

* **Período:** 25/02/2020 a 18/11/2023.

* **Descrição:** O dataset original contém informações sobre doenças pré-existentes, data de início de sintomas, idade, gênero, município e o desfecho (óbito ou não). A maioria dos atributos é categórica (texto), com exceção da idade (numérica) e datas.

  

---

  

## ⚙️ Como Executar

  

### 🧪 Via Google Colab

Se você está utilizando o **Google Colab**, não é necessário configurar um ambiente virtual ou seguir os passos de instalação local:

1. Clone o repositório: `!git clone https://github.com/vinimicius/COVID-ML.git`

2. Navegue até o diretório: `%cd COVID-ML`

3. Importe e execute:

```python

from data_utils import run_pipeline

df = run_pipeline()
```
  

### 💻 Via Local (Linux)

Para rodar localmente, siga os passos abaixo para garantir a consistência das dependências:

  

**Clone o repositório:**

```bash

git clone https://github.com/vinimicius/COVID-ML.git

cd COVID-ML

```

  

**Execute o script de setup:**

Este script criará o ambiente virtual (`.venv`) e instalará as bibliotecas necessárias.

```bash

chmod +x setup.sh

./setup.sh

```

  

**Ative o ambiente:**

```bash

source .venv/bin/activate

```

  

## 🛠️ Procedimento de Obtenção de Dados Limpos

O projeto utiliza um motor de processamento modularizado em `data_utils.py`. O procedimento automatizado segue estas etapas:

  

**Extração:** Download e descompactação automática dos dados brutos do SEADE.

  

**Limpeza:** Filtragem de valores nulos e remoção de inconsistências geográficas ou de preenchimento.

  

**Pré-processamento:**

* **Binarização:** Conversão de indicadores de doenças (SIM/NÃO) para valores binários (0 e 1), incluindo correções de codificação de caracteres (*encoding*).

* **Encoding:** Aplicação de *One-Hot Encoding* para a variável de gênero.

* **Escalonamento:** Normalização da coluna `idade` utilizando **MinMaxScaler**, garantindo que a alta sensibilidade do modelo a esta variável seja tratada proporcionalmente aos dados binários.

* **Otimização:** Tipagem híbrida dos dados (`int8` e `float32`) para reduzir o consumo de memória RAM e aumentar a velocidade de treinamento.

  

## 🚀 Tecnologias Utilizadas

* **Python 3.10+**

* **Pandas:** Manipulação de DataFrames.

* **Scikit-Learn:** Pré-processamento, escalonamento e árvores de decisão.

* **Tensorflow:** Redes neurais.

* **Matplotlib & Seaborn:** Visualizações e análises exploratórias.