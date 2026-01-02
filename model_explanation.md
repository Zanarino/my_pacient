# Explicação do Modelo Preditivo de Hospitalização

## 📋 Índice
1. [Visão Geral](#visão-geral)
2. [Tipo de Problema](#tipo-de-problema)
3. [Modelos Utilizados](#modelos-utilizados)
4. [Features (Variáveis Preditoras)](#features-variáveis-preditoras)
5. [Métricas de Avaliação](#métricas-de-avaliação)
6. [Como Interpretar os Resultados](#como-interpretar-os-resultados)
7. [Limitações](#limitações)
8. [Como Usar o Modelo](#como-usar-o-modelo)

---

## 🎯 Visão Geral

Este projeto desenvolve **modelos preditivos** para estimar a probabilidade de **hospitalização** de pacientes idosos em dois horizontes temporais:

- **Modelo 1**: Prediz hospitalização no **próximo ano** (12 meses)
- **Modelo 2**: Prediz hospitalização nos **próximos 3 anos** (36 meses)

### Objetivo Clínico
Identificar pacientes em **alto risco de hospitalização** para permitir:
- Intervenções preventivas precoces
- Alocação eficiente de recursos de saúde
- Monitoramento mais próximo de pacientes de risco
- Planejamento de cuidados personalizados

---

## 🔬 Tipo de Problema

### Classificação Binária Supervisionada

**O que é?**
- **Classificação**: Atribuir uma categoria (sim/não) a cada paciente
- **Binária**: Apenas duas categorias possíveis (hospitalizado ou não hospitalizado)
- **Supervisionada**: Aprendemos com exemplos históricos onde já sabemos o resultado

**Como funciona?**
1. O modelo aprende padrões nos dados históricos de pacientes
2. Identifica características que diferenciam pacientes hospitalizados dos não hospitalizados
3. Usa esses padrões para prever novos casos

**Exemplo prático:**
```
Paciente A: 82 anos, frágil, 5 comorbidades, baixa mobilidade
→ Modelo prevê: ALTO RISCO (85% probabilidade de hospitalização)

Paciente B: 73 anos, não frágil, 1 comorbidade, boa mobilidade
→ Modelo prevê: BAIXO RISCO (15% probabilidade de hospitalização)
```

---

## 🤖 Modelos Utilizados

Testamos **5 algoritmos diferentes** para encontrar o melhor desempenho:

### 1. Logistic Regression (Regressão Logística)

**O que é?**
- Modelo estatístico linear que estima probabilidades
- Um dos modelos mais simples e interpretáveis

**Como funciona?**
- Calcula uma pontuação ponderada das características do paciente
- Converte essa pontuação em probabilidade (0% a 100%)

**Vantagens:**
- ✅ Muito interpretável (podemos ver o peso de cada fator)
- ✅ Rápido de treinar
- ✅ Funciona bem com datasets pequenos
- ✅ Menos propenso a overfitting

**Desvantagens:**
- ❌ Assume relações lineares (pode perder padrões complexos)
- ❌ Pode ter performance inferior em dados muito complexos

**Por que escolhemos?**
É nosso **modelo baseline** - se modelos mais complexos não superarem este, não valem a complexidade adicional.

---

### 2. Decision Tree (Árvore de Decisão)

**O que é?**
- Modelo baseado em regras do tipo "se-então"
- Cria uma árvore de decisões sequenciais

**Como funciona?**
```
Se idade > 80 anos:
    Se fragilidade = "Frail":
        Se comorbidades > 5:
            → ALTO RISCO
        Senão:
            → RISCO MODERADO
    Senão:
        → BAIXO RISCO
```

**Vantagens:**
- ✅ Muito fácil de interpretar e visualizar
- ✅ Captura relações não-lineares
- ✅ Não requer normalização de dados
- ✅ Identifica automaticamente interações entre variáveis

**Desvantagens:**
- ❌ Propenso a overfitting (memorizar os dados de treino)
- ❌ Instável (pequenas mudanças nos dados podem mudar a árvore)

**Por que escolhemos?**
Fornece **regras clínicas interpretáveis** que médicos podem entender facilmente.

---

### 3. Random Forest (Floresta Aleatória)

**O que é?**
- Ensemble (conjunto) de múltiplas árvores de decisão
- Cada árvore "vota" e a maioria decide

**Como funciona?**
1. Cria 100 árvores de decisão diferentes
2. Cada árvore usa uma amostra aleatória dos dados
3. Cada árvore vota na predição
4. A predição final é a média/maioria dos votos

**Vantagens:**
- ✅ Muito robusto e estável
- ✅ Reduz overfitting comparado a uma única árvore
- ✅ Lida bem com features categóricas e numéricas
- ✅ Fornece importância das features
- ✅ Geralmente boa performance "out-of-the-box"

**Desvantagens:**
- ❌ Menos interpretável que uma única árvore
- ❌ Mais lento para treinar e prever

**Por que escolhemos?**
É um dos **modelos mais confiáveis** em machine learning médico, balanceando performance e robustez.

---

### 4. Gradient Boosting (Boosting Gradiente)

**O que é?**
- Ensemble sequencial de árvores
- Cada nova árvore corrige os erros da anterior

**Como funciona?**
1. Treina uma árvore simples
2. Identifica onde ela errou
3. Treina uma nova árvore focada nesses erros
4. Repete 100 vezes
5. Combina todas as árvores (cada uma com um peso)

**Vantagens:**
- ✅ Geralmente a melhor performance
- ✅ Captura padrões muito complexos
- ✅ Fornece importância das features

**Desvantagens:**
- ❌ Alto risco de overfitting em datasets pequenos
- ❌ Mais difícil de interpretar
- ❌ Requer ajuste cuidadoso de hiperparâmetros

**Por que escolhemos?**
Pode alcançar a **melhor performance**, mas requer validação cuidadosa para evitar overfitting.


---

## 📊 Features (Variáveis Preditoras)

### Features Demográficas

| Feature | Descrição | Tipo | Importância Esperada |
|---------|-----------|------|---------------------|
| `age` | Idade do paciente (70-85 anos) | Numérica | ⭐⭐⭐ Alta |
| `gender` | Gênero (M/F) | Categórica | ⭐ Baixa |
| `age_group` | Grupo etário (70-74, 75-79, 80+) | Categórica | ⭐⭐ Média |

### Features de Fragilidade e Funcionalidade

| Feature | Descrição | Tipo | Importância Esperada |
|---------|-----------|------|---------------------|
| `fried` | Status de fragilidade (Non frail, Pre-frail, Frail) | Categórica | ⭐⭐⭐⭐⭐ Muito Alta |
| `katz_index` | Índice de independência em atividades básicas (0-6) | Numérica | ⭐⭐⭐⭐ Alta |
| `iadl_grade` | Atividades instrumentais da vida diária (0-31) | Numérica | ⭐⭐⭐⭐ Alta |
| `functional_independence_score` | Score combinado Katz + IADL | Numérica | ⭐⭐⭐⭐ Alta |

**Interpretação:**
- **Fried**: Critério padrão-ouro de fragilidade. Pacientes "Frail" têm risco muito maior.
- **Katz**: Mede capacidade de fazer atividades básicas (banho, vestir-se, etc.)
- **IADL**: Mede atividades mais complexas (cozinhar, gerenciar finanças, etc.)

### Features de Mobilidade e Desempenho Físico

| Feature | Descrição | Tipo | Importância Esperada |
|---------|-----------|------|---------------------|
| `gait_speed_4m` | Velocidade da marcha em 4 metros (m/s) | Numérica | ⭐⭐⭐⭐ Alta |
| `raise_chair_time` | Tempo para levantar da cadeira (segundos) | Numérica | ⭐⭐⭐ Média-Alta |
| `balance_single` | Equilíbrio em pé único | Categórica | ⭐⭐ Média |
| `falls_one_year` | Número de quedas no último ano | Numérica | ⭐⭐⭐⭐ Alta |
| `frailty_physical_score` | Score combinado de mobilidade | Numérica | ⭐⭐⭐⭐ Alta |

**Interpretação:**
- **Gait speed**: Velocidade < 0.8 m/s indica fragilidade
- **Falls**: Quedas são forte preditor de hospitalização

### Features de Comorbidades e Medicações

| Feature | Descrição | Tipo | Importância Esperada |
|---------|-----------|------|---------------------|
| `comorbidities_count` | Número total de comorbidades | Numérica | ⭐⭐⭐⭐⭐ Muito Alta |
| `comorbidities_significant_count` | Comorbidades graves | Numérica | ⭐⭐⭐⭐ Alta |
| `comorbidities_most_important` | Comorbidade principal | Categórica | ⭐⭐⭐ Média-Alta |
| `medication_count` | Número de medicamentos | Numérica | ⭐⭐⭐⭐ Alta |
| `medication_per_comorbidity` | Razão medicamentos/comorbidades | Numérica | ⭐⭐⭐ Média-Alta |

**Interpretação:**
- **Multimorbidade**: Múltiplas comorbidades aumentam risco exponencialmente
- **Polifarmácia**: Muitos medicamentos indicam complexidade clínica

### Features Cognitivas e Psicológicas

| Feature | Descrição | Tipo | Importância Esperada |
|---------|-----------|------|---------------------|
| `mmse_total_score` | Mini-Mental State Exam (0-30) | Numérica | ⭐⭐⭐⭐ Alta |
| `depression_total_score` | Score de depressão | Numérica | ⭐⭐⭐ Média-Alta |
| `cognitive_psych_risk` | Score combinado cognitivo-psicológico | Numérica | ⭐⭐⭐⭐ Alta |
| `anxiety_perception` | Percepção de ansiedade | Numérica | ⭐⭐ Média |

**Interpretação:**
- **MMSE < 24**: Indica comprometimento cognitivo
- **Depressão**: Associada a piores outcomes de saúde

### Features de Estilo de Vida

| Feature | Descrição | Tipo | Importância Esperada |
|---------|-----------|------|---------------------|
| `smoking` | Status de tabagismo | Categórica | ⭐⭐ Média |
| `alcohol_units` | Unidades de álcool por semana | Numérica | ⭐ Baixa-Média |
| `activity_regular` | Nível de atividade física regular | Categórica | ⭐⭐⭐ Média-Alta |
| `bmi_score` | Índice de Massa Corporal | Numérica | ⭐⭐ Média |

### Features Sociais

| Feature | Descrição | Tipo | Importância Esperada |
|---------|-----------|------|---------------------|
| `living_alone` | Mora sozinho (sim/não) | Categórica | ⭐⭐⭐ Média-Alta |
| `social_visits` | Frequência de visitas sociais | Numérica | ⭐⭐ Média |
| `social_calls` | Frequência de ligações | Numérica | ⭐ Baixa-Média |

**Interpretação:**
- **Isolamento social**: Morar sozinho pode aumentar risco

---

## 📈 Métricas de Avaliação

### 1. Accuracy (Acurácia)

**O que é?**
Proporção de predições corretas sobre o total.

**Fórmula:**
```
Accuracy = (Acertos) / (Total de predições)
         = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretação:**
- 0.80 = 80% de acertos
- **CUIDADO**: Pode ser enganosa em classes desbalanceadas!

**Exemplo:**
Se 90% dos pacientes NÃO são hospitalizados, um modelo que sempre prediz "não hospitalizado" teria 90% de accuracy, mas seria inútil!

---

### 2. Precision (Precisão)

**O que é?**
Dos pacientes que o modelo previu como "serão hospitalizados", quantos realmente foram?

**Fórmula:**
```
Precision = TP / (TP + FP)
```

**Interpretação:**
- 0.75 = 75% dos pacientes preditos como "alto risco" realmente foram hospitalizados
- Alta precision = Poucos falsos alarmes

**Quando é importante?**
Quando o custo de **falsos positivos** é alto (ex: intervenções caras/invasivas)

---

### 3. Recall / Sensitivity (Sensibilidade)

**O que é?**
Dos pacientes que realmente foram hospitalizados, quantos o modelo conseguiu identificar?

**Fórmula:**
```
Recall = TP / (TP + FN)
```

**Interpretação:**
- 0.85 = 85% dos pacientes hospitalizados foram corretamente identificados
- Alto recall = Poucos casos perdidos

**Quando é importante?**
Quando o custo de **falsos negativos** é alto (ex: não identificar paciente de alto risco)

---

### 4. F1-Score

**O que é?**
Média harmônica entre Precision e Recall. Balanceia ambas as métricas.

**Fórmula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretação:**
- Varia de 0 a 1
- Quanto maior, melhor
- Útil quando queremos balancear precision e recall

---

### 5. ROC-AUC (Area Under ROC Curve)

**O que é?**
Mede a capacidade do modelo de discriminar entre classes em todos os thresholds possíveis.

**Interpretação:**
- **1.0**: Classificador perfeito
- **0.9-1.0**: Excelente
- **0.8-0.9**: Muito bom
- **0.7-0.8**: Bom
- **0.6-0.7**: Razoável
- **0.5-0.6**: Ruim
- **0.5**: Aleatório (jogar moeda)
- **< 0.5**: Pior que aleatório

**Por que é importante?**
- Não depende de um threshold específico
- Funciona bem com classes desbalanceadas
- É nossa **métrica principal** para comparar modelos

---

### 6. Confusion Matrix (Matriz de Confusão)

**O que é?**
Tabela que mostra todos os tipos de acertos e erros.

```
                    Predito
                 Não Hosp.  Hosp.
Real  Não Hosp.     TN       FP
      Hosp.         FN       TP
```

**Componentes:**
- **TP (True Positive)**: Previu hospitalização ✅ e estava certo ✅
- **TN (True Negative)**: Previu não hospitalização ✅ e estava certo ✅
- **FP (False Positive)**: Previu hospitalização ❌ mas não aconteceu (Falso Alarme)
- **FN (False Negative)**: Previu não hospitalização ❌ mas aconteceu (Caso Perdido)

**Exemplo:**
```
                Predito
             Não Hosp.  Hosp.
Real  Não H.    25        3     ← 3 falsos alarmes
      Hosp.      2        6     ← 2 casos perdidos
```

---

## 🔍 Como Interpretar os Resultados

### Comparando Modelos

**Critérios de seleção:**

1. **ROC-AUC no teste** (métrica principal)
   - Escolher o modelo com maior AUC
   - Verificar que AUC > 0.7 (mínimo aceitável)

2. **Gap Treino-Teste** (verificar overfitting)
   - Gap < 0.10: Excelente generalização ✅
   - Gap 0.10-0.15: Leve overfitting ⚡
   - Gap > 0.15: Overfitting preocupante ❌

3. **F1-Score** (balanceamento)
   - Importante quando classes são desbalanceadas

4. **Recall** (se não podemos perder casos)
   - Priorizar se o custo de não identificar paciente de risco é alto

### Interpretando Feature Importance

**Top 3 features mais importantes** geralmente indicam:

1. **Fragilidade (fried)**: Preditor mais forte
2. **Comorbidades**: Número e gravidade
3. **Mobilidade**: Velocidade da marcha, quedas

**Como usar:**
- Features importantes devem fazer sentido clínico
- Se features estranhas aparecem no topo, pode indicar problemas nos dados

### Usando o Modelo na Prática

**Exemplo de uso clínico:**

```python
# Paciente novo
paciente = {
    'age': 82,
    'fried': 'Frail',
    'comorbidities_count': 7,
    'gait_speed_4m': 0.6,
    'falls_one_year': 2,
    ...
}

# Predição
probabilidade = modelo.predict_proba(paciente)[0][1]

if probabilidade > 0.7:
    print("ALTO RISCO - Intervenção recomendada")
elif probabilidade > 0.4:
    print("RISCO MODERADO - Monitoramento próximo")
else:
    print("BAIXO RISCO - Acompanhamento padrão")
```

---

## ⚠️ Limitações

### 1. Tamanho do Dataset

**Problema:**
- Apenas 117 observações
- Dataset muito pequeno para machine learning

**Impacto:**
- ❌ Modelos podem não generalizar bem
- ❌ Métricas podem variar significativamente
- ❌ Risco alto de overfitting
- ❌ Difícil capturar padrões raros

**Mitigação:**
- ✅ Usamos validação cruzada
- ✅ Regularização forte nos modelos
- ✅ Modelos mais simples (menos propensos a overfitting)

### 2. Desbalanceamento de Classes

**Problema:**
- Hospitalização 1 ano: ~24% (desbalanceado)
- Hospitalização 3 anos: ~60% (mais balanceado)

**Impacto:**
- ❌ Modelo pode tender a prever classe majoritária
- ❌ Accuracy pode ser enganosa

**Mitigação:**
- ✅ Class weights balanceados
- ✅ Métricas apropriadas (F1, ROC-AUC)
- ✅ Stratified sampling

### 3. Generalização

**Problema:**
- Modelo treinado em uma população específica
- Pode não funcionar bem em outras populações

**Recomendação:**
- ⚠️ Validar em novos dados antes de uso clínico
- ⚠️ Re-treinar periodicamente com novos dados

### 4. Causalidade vs Correlação

**Problema:**
- Modelo identifica correlações, não causas
- Não podemos afirmar que X causa Y

**Exemplo:**
- Modelo pode identificar que "morar sozinho" está associado a hospitalização
- Mas não significa que morar sozinho CAUSA hospitalização
- Pode haver fatores confundidores

### 5. Dados Ausentes

**Problema:**
- Algumas features têm muitos valores ausentes
- Imputação pode introduzir viés

**Mitigação:**
- ✅ Usamos imputação com mediana/moda
- ✅ Removemos features com >50% missing

---

## 🚀 Como Usar o Modelo

### Instalação

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Verificar que o dataset está em:
#    raw_data/Virtual_Patient_Models_Dataset.csv
```

### Execução

```bash
# Executar o pipeline completo
python predictive_model.py
```

### Outputs Gerados

**Métricas:**
- `outputs/model_comparison_1year.csv`: Comparação de modelos para 1 ano
- `outputs/model_comparison_3years.csv`: Comparação de modelos para 3 anos

**Visualizações:**
- `outputs/confusion_matrix_*.png`: Matrizes de confusão
- `outputs/roc_curve_*.png`: Curvas ROC
- `outputs/feature_importance_*.png`: Importância das features

**Modelos Salvos:**
- `models/*.pkl`: Modelos treinados para uso futuro

### Usando um Modelo Salvo

```python
import pickle
import pandas as pd

# Carregar modelo
with open('models/random_forest_1year.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Carregar scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Preparar dados do novo paciente
# (seguir mesmas transformações do treino)
novo_paciente = preparar_features(dados_paciente)
novo_paciente_scaled = scaler.transform(novo_paciente)

# Predição
probabilidade = modelo.predict_proba(novo_paciente_scaled)[0][1]
print(f"Probabilidade de hospitalização: {probabilidade:.1%}")
```

---

## 📚 Referências e Leituras Recomendadas

### Fragilidade em Idosos
- Fried LP, et al. (2001). "Frailty in older adults: evidence for a phenotype"
- Clegg A, et al. (2013). "Frailty in elderly people"

### Machine Learning em Saúde
- Rajkomar A, et al. (2019). "Machine Learning in Medicine"
- Beam AL, Kohane IS. (2018). "Big Data and Machine Learning in Health Care"

### Métricas de Avaliação
- Saito T, Rehmsmeier M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot"

---

## 💡 Conclusão

Este modelo fornece uma **ferramenta de apoio à decisão clínica** para identificar pacientes idosos em risco de hospitalização. 

**Pontos-chave:**
- ✅ Usa múltiplos algoritmos para robustez
- ✅ Altamente interpretável (feature importance)
- ✅ Métricas apropriadas para avaliação
- ⚠️ Limitado por tamanho do dataset
- ⚠️ Requer validação clínica antes de uso

**Próximos passos recomendados:**
1. Coletar mais dados para melhorar robustez
2. Validar em população externa
3. Integrar com sistema de prontuário eletrônico
4. Desenvolver interface para uso clínico
5. Monitorar performance ao longo do tempo

---

**Desenvolvido com ❤️ para melhorar o cuidado de pacientes idosos**

