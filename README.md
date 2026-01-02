<<<<<<< HEAD
# :health_worker: my_pacient

Facilitate the monitoring of elderly patients, with analysis of correlation between physical and psychological parameters, limitations, balance, depression and coginition, we can outline better treatment plans for this patients.

Make a model to try to predict possible problems like:
- Which patients may be at a higher risk of needing regular hospitalization :ambulance:
- Have a deterioration of physical function :chart_with_downwards_trend:

This model utilizing machine learning algorithms to produce real-time risk scores for each patient.
=======
# Modelo Preditivo de Hospitalização de Pacientes Idosos

Este projeto implementa modelos de Machine Learning para prever a probabilidade de hospitalização de pacientes idosos em dois horizontes temporais: **1 ano** e **3 anos**.

## 🎯 Objetivo

Identificar pacientes em alto risco de hospitalização para permitir:
- Intervenções preventivas precoces
- Alocação eficiente de recursos de saúde
- Monitoramento personalizado
- Planejamento de cuidados

## 📊 Modelos Implementados

- **Logistic Regression**: Baseline interpretável
- **Decision Tree**: Regras clínicas simples
- **Random Forest**: Ensemble robusto
- **Gradient Boosting**: Modelo avançado
- **XGBoost**: Estado da arte (se disponível)

## 🚀 Como Usar

### 1. Instalação

```bash
# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparar Dados

O dataset deve estar em: `raw_data/Virtual_Patient_Models_Dataset.csv`

**Opções:**

**A) Se você já tem o arquivo CSV:**
```bash
# Coloque o arquivo em raw_data/
cp seu_arquivo.csv raw_data/Virtual_Patient_Models_Dataset.csv
```

**B) Extrair do notebook:**
```bash
# Executar script de extração
python extract_data.py
```

**C) Verificar disponibilidade:**
```bash
python check_dataset.py
```

### 3. Executar o Modelo

```bash
# Executar pipeline completo
python predictive_model.py
```

## 📁 Estrutura do Projeto

```
my_pacient/
├── raw_data/                          # Dados brutos
│   └── Virtual_Patient_Models_Dataset.csv
├── outputs/                           # Resultados gerados
│   ├── model_comparison_1year.csv
│   ├── model_comparison_3years.csv
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── feature_importance_*.png
├── models/                            # Modelos treinados
│   ├── *.pkl
│   └── scaler.pkl
├── predictive_model.py                # Script principal
├── model_explanation.md               # Documentação detalhada
├── extract_data.py                    # Extração de dados
├── check_dataset.py                   # Verificação de dados
├── requirements.txt                   # Dependências
└── README.md                          # Este arquivo
```

## 📈 Outputs Gerados

### Métricas
- `model_comparison_*.csv`: Comparação de performance dos modelos

### Visualizações
- `confusion_matrix_*.png`: Matrizes de confusão
- `roc_curve_*.png`: Curvas ROC com AUC
- `feature_importance_*.png`: Importância das variáveis

### Modelos Salvos
- `models/*.pkl`: Modelos treinados para uso futuro

## 📚 Documentação

Para entender em detalhes:
- **Como os modelos funcionam**: Veja `model_explanation.md`
- **Métricas de avaliação**: Veja seção de métricas em `model_explanation.md`
- **Features utilizadas**: Veja seção de features em `model_explanation.md`
- **Limitações**: Veja seção de limitações em `model_explanation.md`

## 🔍 Principais Features

### Demográficas
- Idade, gênero

### Fragilidade
- Status de fragilidade (Fried)
- Índices funcionais (Katz, IADL)

### Mobilidade
- Velocidade da marcha
- Tempo para levantar da cadeira
- Histórico de quedas

### Clínicas
- Número de comorbidades
- Número de medicamentos
- Comorbidades significativas

### Cognitivas/Psicológicas
- MMSE (cognição)
- Score de depressão
- Ansiedade

### Estilo de Vida
- Atividade física
- Tabagismo
- Consumo de álcool

## ⚠️ Limitações Importantes

1. **Dataset pequeno** (117 observações) - risco de overfitting
2. **Classe desbalanceada** para hospitalização 1 ano (~24%)
3. **Generalização limitada** - validar em novas populações
4. **Correlação ≠ Causalidade** - modelo não identifica causas

## 📊 Métricas Principais

- **ROC-AUC**: Capacidade de discriminação (métrica principal)
- **F1-Score**: Balanço entre precision e recall
- **Precision**: Dos preditos como alto risco, quantos realmente são
- **Recall**: Dos realmente em risco, quantos identificamos

## 🔧 Uso Avançado

### Carregar Modelo Salvo

```python
import pickle
import pandas as pd

# Carregar modelo
with open('models/random_forest_1year.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Carregar scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Fazer predição
novo_paciente = preparar_features(dados)
novo_paciente_scaled = scaler.transform(novo_paciente)
probabilidade = modelo.predict_proba(novo_paciente_scaled)[0][1]

print(f"Risco de hospitalização: {probabilidade:.1%}")
```

## 🤝 Contribuindo

Para melhorar este projeto:
1. Coletar mais dados para aumentar robustez
2. Validar em populações externas
3. Adicionar novos modelos
4. Melhorar feature engineering
5. Desenvolver interface de uso clínico

## 📝 Licença

Este projeto é para fins educacionais e de pesquisa.

## 👥 Autor

Data Science Team - 2026

---

**⚕️ Desenvolvido para melhorar o cuidado de pacientes idosos**
>>>>>>> eda
