# Relatório de Análise: Modelo Preditivo de Hospitalização
**Data de Geração**: 02/01/2026 16:59:22
**Autor**: Rafael Zanarino
---

## 📋 Sumário Executivo

Este relatório apresenta os resultados da modelagem preditiva de hospitalização de pacientes idosos em dois horizontes temporais: **1 ano** e **3 anos**.

### 🏆 Melhores Modelos

- **Hospitalização 1 ano**: Gradient Boosting (ROC-AUC: 0.816)
- **Hospitalização 3 anos**: Gradient Boosting (ROC-AUC: 0.537)

---

## 🎯 Análise: Predição de Hospitalização em 1 Ano

### Comparação de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overfitting |
|--------|----------|-----------|--------|----------|---------|-------------|
| Logistic Regression | 0.750 | 0.796 | 0.750 | 0.768 | 0.774 | ⚠️ 0.220 |
| Decision Tree | 0.583 | 0.709 | 0.583 | 0.631 | 0.575 | ⚠️ 0.408 |
| Random Forest | 0.833 | 0.795 | 0.833 | 0.802 | 0.783 | ⚠️ 0.215 |
| Gradient Boosting | 0.889 | 0.875 | 0.889 | 0.865 | 0.816 | ⚠️ 0.184 |

### 🔍 Análise do Melhor Modelo: Gradient Boosting

**Por que este modelo foi escolhido?**

O **Gradient Boosting** apresentou o melhor desempenho com ROC-AUC de **0.816**, indicando muito boa capacidade de discriminação entre pacientes que serão e não serão hospitalizados.

**Métricas de Performance:**

- **Accuracy**: 88.9% - Proporção de predições corretas
- **Precision**: 87.5% - Dos preditos como 'alto risco', 87.5% realmente foram hospitalizados
- **Recall**: 88.9% - Dos pacientes hospitalizados, 88.9% foram corretamente identificados
- **F1-Score**: 0.865 - Balanço entre precision e recall

⚠️ **Overfitting Detectado**: Gap de 0.184 indica que o modelo pode estar memorizando os dados de treino.

---

## 🎯 Análise: Predição de Hospitalização em 3 Anos

### Comparação de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overfitting |
|--------|----------|-----------|--------|----------|---------|-------------|
| Logistic Regression | 0.306 | 0.354 | 0.306 | 0.321 | 0.412 | ⚠️ 0.527 |
| Decision Tree | 0.250 | 0.418 | 0.250 | 0.296 | 0.431 | ⚠️ 0.366 |
| Random Forest | 0.417 | 0.377 | 0.417 | 0.395 | 0.411 | ⚠️ 0.532 |
| Gradient Boosting | 0.472 | 0.413 | 0.472 | 0.437 | 0.537 | ⚠️ 0.463 |

### 🔍 Análise do Melhor Modelo: Gradient Boosting

**Por que este modelo foi escolhido?**

O **Gradient Boosting** apresentou o melhor desempenho com ROC-AUC de **0.537**, indicando razoável capacidade de discriminação.

**Métricas de Performance:**

- **Accuracy**: 47.2%
- **Precision**: 41.3%
- **Recall**: 47.2%
- **F1-Score**: 0.437

---

## 📊 Explicação das Visualizações

### 1. Matrizes de Confusão

**Arquivos**: `confusion_matrix_1year.png` e `confusion_matrix_3years.png`

**O que mostram:**

As matrizes de confusão visualizam os acertos e erros de cada modelo:

```
                Predito
             Não Hosp.  Hosp.
Real  Não H.    TN       FP     ← Falsos Alarmes
      Hosp.     FN       TP     ← Casos Perdidos
```

- **TN (True Negative)**: Pacientes corretamente identificados como baixo risco
- **TP (True Positive)**: Pacientes corretamente identificados como alto risco
- **FP (False Positive)**: Falsos alarmes - preditos como alto risco mas não hospitalizados
- **FN (False Negative)**: Casos perdidos - não identificados mas foram hospitalizados

**Como interpretar**: Quanto maior os valores na diagonal (TN e TP), melhor o modelo.

### 2. Curvas ROC

**Arquivos**: `roc_curve_1year.png` e `roc_curve_3years.png`

**O que mostram:**

As curvas ROC (Receiver Operating Characteristic) mostram o trade-off entre:
- **True Positive Rate (Recall)**: Taxa de acerto nos casos positivos
- **False Positive Rate**: Taxa de falsos alarmes

**Interpretação da AUC (Area Under Curve)**:
- **0.9 - 1.0**: Excelente discriminação
- **0.8 - 0.9**: Muito boa discriminação
- **0.7 - 0.8**: Boa discriminação
- **0.6 - 0.7**: Razoável
- **0.5**: Aleatório (jogar moeda)

**Como interpretar**: Quanto mais próxima a curva do canto superior esquerdo, melhor o modelo.

### 3. Importância das Features

**Arquivos**: `feature_importance_1year.png` e `feature_importance_3years.png`

**O que mostram:**

Estes gráficos mostram quais variáveis têm maior influência nas predições dos modelos baseados em árvores (Decision Tree, Random Forest, Gradient Boosting).

**Como interpretar**:
- Features no topo da lista têm maior impacto nas predições
- Ajuda a entender quais fatores clínicos são mais relevantes
- Útil para validação clínica (as features importantes fazem sentido médico?)

---

## 💡 Conclusões e Recomendações

### Principais Achados

1. **Modelo mais eficaz para 1 ano**: Gradient Boosting com AUC de 0.816
2. **Modelo mais eficaz para 3 anos**: Gradient Boosting com AUC de 0.537
3. **Predição de curto prazo** (1 ano) apresentou melhor performance que longo prazo (3 anos)

### Recomendações de Uso

**Para uso clínico:**

1. Utilizar o **Gradient Boosting** para identificar pacientes em risco de hospitalização no próximo ano
2. Utilizar o **Gradient Boosting** para planejamento de cuidados de longo prazo
3. Considerar intervenções preventivas para pacientes identificados como alto risco
4. Monitorar continuamente a performance dos modelos com novos dados

### Limitações

⚠️ **Importante considerar:**

1. **Dataset pequeno** (117 observações) - limita a confiabilidade estatística
2. **Validação externa necessária** - testar em nova população antes de uso clínico
3. **Modelos não substituem julgamento clínico** - usar como ferramenta de apoio à decisão
4. **Re-treinamento periódico** - atualizar modelos com novos dados regularmente

### Próximos Passos

1. ✅ Coletar mais dados para aumentar robustez
2. ✅ Validar em população externa
3. ✅ Desenvolver interface de uso clínico
4. ✅ Implementar monitoramento contínuo de performance
5. ✅ Realizar estudos de impacto clínico

---

**Relatório gerado automaticamente pelo HospitalizationPredictor**
