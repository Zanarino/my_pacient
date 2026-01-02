"""
Modelo Preditivo de Hospitalização de Pacientes Idosos
========================================================

Este script implementa modelos de Machine Learning para prever a probabilidade
de hospitalização de pacientes idosos em dois horizontes temporais:
- 1 ano (hospitalization_one_year)
- 3 anos (hospitalization_three_years)

Autor: Rafael Zanarino
Data: 2026-01-01
"""

# ============================================================================
# IMPORTAÇÕES
# ============================================================================

# Manipulação de dados
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suprimir warnings para output mais limpo

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: Pré-processamento
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Scikit-learn: Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Scikit-learn: Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Tratamento de desbalanceamento
from imblearn.over_sampling import SMOTE

# Utilitários
import os
import pickle
from datetime import datetime


# ============================================================================
# CLASSE PRINCIPAL: HospitalizationPredictor
# ============================================================================

class HospitalizationPredictor:
    """
    Classe para prever hospitalização de pacientes idosos.
    
    Esta classe encapsula todo o pipeline de machine learning:
    1. Carregamento e limpeza de dados
    2. Feature engineering
    3. Treinamento de modelos
    4. Avaliação e visualização
    5. Interpretação de resultados
    
    Attributes:
        data (pd.DataFrame): Dataset original
        X_train, X_test: Features de treino e teste
        y_train_1y, y_test_1y: Targets de 1 ano
        y_train_3y, y_test_3y: Targets de 3 anos
        models_1y (dict): Modelos treinados para 1 ano
        models_3y (dict): Modelos treinados para 3 anos
        scaler (StandardScaler): Normalizador de features
        feature_names (list): Nomes das features após processamento
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa o preditor.
        
        Args:
            random_state (int): Seed para reprodutibilidade dos resultados
        """
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train_1y = None
        self.y_test_1y = None
        self.y_train_3y = None
        self.y_test_3y = None
        self.models_1y = {}
        self.models_3y = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.label_encoders = {}  # Armazena encoders para variáveis categóricas
        
        # Criar diretórios para outputs se não existirem
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        print("✅ HospitalizationPredictor inicializado com sucesso!")
        print(f"📊 Random state: {self.random_state}")
        print(f"📁 Diretórios criados: outputs/, models/\n")
    
    
    def load_data(self, filepath='raw_data/Virtual_Patient_Models_Dataset.csv'):
        """
        Carrega o dataset de pacientes.
        
        Args:
            filepath (str): Caminho para o arquivo CSV
            
        Returns:
            pd.DataFrame: Dataset carregado
        """
        print("=" * 70)
        print("📂 CARREGANDO DADOS")
        print("=" * 70)
        
        try:
            self.data = pd.read_csv(filepath)
            print(f"✅ Dataset carregado com sucesso!")
            print(f"   📊 Shape: {self.data.shape}")
            print(f"   👥 Número de pacientes únicos: {self.data['part_id'].nunique()}")
            print(f"   📋 Número de observações: {len(self.data)}")
            print(f"   📈 Features disponíveis: {self.data.shape[1]}")
            
            # Informações sobre os targets
            print(f"\n🎯 DISTRIBUIÇÃO DOS TARGETS:")
            print(f"   Hospitalização 1 ano:")
            print(f"      - Sim: {self.data['hospitalization_one_year'].sum()} ({self.data['hospitalization_one_year'].mean()*100:.1f}%)")
            print(f"      - Não: {(self.data['hospitalization_one_year']==0).sum()} ({(1-self.data['hospitalization_one_year'].mean())*100:.1f}%)")
            
            print(f"   Hospitalização 3 anos:")
            print(f"      - Sim: {self.data['hospitalization_three_years'].sum()} ({self.data['hospitalization_three_years'].mean()*100:.1f}%)")
            print(f"      - Não: {(self.data['hospitalization_three_years']==0).sum()} ({(1-self.data['hospitalization_three_years'].mean())*100:.1f}%)")
            
            return self.data
            
        except FileNotFoundError:
            print(f"❌ ERRO: Arquivo não encontrado: {filepath}")
            print(f"   Por favor, certifique-se de que o arquivo existe no caminho especificado.")
            raise
        except Exception as e:
            print(f"❌ ERRO ao carregar dados: {str(e)}")
            raise
    
    
    def prepare_features(self):
        """
        Prepara as features para o modelo.
        
        Este método realiza:
        1. Seleção de features relevantes
        2. Tratamento de valores ausentes
        3. Encoding de variáveis categóricas
        4. Criação de novas features (feature engineering)
        5. Normalização de features numéricas
        
        Returns:
            tuple: (X, y_1year, y_3years) - Features e targets preparados
        """
        print("\n" + "=" * 70)
        print("🔧 PREPARAÇÃO DE FEATURES")
        print("=" * 70)
        
        df = self.data.copy()
        
        # ====================================================================
        # 1. REMOVER COLUNAS NÃO PREDITIVAS
        # ====================================================================
        print("\n1️⃣ Removendo colunas não preditivas...")
        
        # Colunas a serem removidas:
        # - Identificadores: part_id, clinical_visit
        # - Data: q_date (não é preditiva diretamente)
        # - Targets: hospitalization_one_year, hospitalization_three_years
        columns_to_drop = [
            'part_id',           # ID do paciente (não preditivo)
            'clinical_visit',    # Número da visita (não preditivo)
            'q_date',            # Data da consulta (não preditiva)
            'hospitalization_one_year',    # Target 1
            'hospitalization_three_years'  # Target 2
        ]
        
        # Salvar targets antes de remover
        y_1year = df['hospitalization_one_year'].copy()
        y_3years = df['hospitalization_three_years'].copy()
        
        # Remover colunas
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"   ✅ {len(columns_to_drop)} colunas removidas")
        print(f"   📊 Shape após remoção: {df.shape}")
        
        # ====================================================================
        # 2. IDENTIFICAR TIPOS DE FEATURES
        # ====================================================================
        print("\n2️⃣ Identificando tipos de features...")
        
        # Features numéricas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Features categóricas
        categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        print(f"   📊 Features numéricas: {len(numeric_features)}")
        print(f"   📝 Features categóricas: {len(categorical_features)}")
        
        # ====================================================================
        # 3. TRATAMENTO DE VALORES AUSENTES E OUTLIERS
        # ====================================================================
        print("\n3️⃣ Tratando valores ausentes e outliers...")
        
        # Valores como 999 geralmente indicam "missing" em datasets médicos
        # Substituir por NaN para tratamento adequado
        df = df.replace(999, np.nan)
        
        # Contar missing values antes do tratamento
        missing_before = df.isnull().sum().sum()
        print(f"   ⚠️ Missing values detectados: {missing_before}")
        
        # Imputação para features numéricas (usar mediana - mais robusta a outliers)
        if numeric_features:
            imputer_num = SimpleImputer(strategy='median')
            df[numeric_features] = imputer_num.fit_transform(df[numeric_features])
            print(f"   ✅ Features numéricas imputadas com mediana")
        
        # Imputação para features categóricas (usar moda - valor mais frequente)
        if categorical_features:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])
            print(f"   ✅ Features categóricas imputadas com moda")
        
        missing_after = df.isnull().sum().sum()
        print(f"   ✅ Missing values após tratamento: {missing_after}")
        
        # ====================================================================
        # 4. FEATURE ENGINEERING
        # ====================================================================
        print("\n4️⃣ Criando novas features (Feature Engineering)...")
        
        # Feature 1: Razão medicamentos por comorbidade
        # Indica se o paciente está adequadamente medicado para suas condições
        df['medication_per_comorbidity'] = df['medication_count'] / (df['comorbidities_count'] + 1)
        print(f"   ✅ Criada: medication_per_comorbidity")
        
        # Feature 2: Score de fragilidade combinado
        # Combina múltiplos indicadores de fragilidade física
        if 'gait_speed_4m' in df.columns and 'raise_chair_time' in df.columns:
            # Normalizar para escala 0-1 (quanto maior, mais frágil)
            gait_normalized = (df['gait_speed_4m'] - df['gait_speed_4m'].min()) / (df['gait_speed_4m'].max() - df['gait_speed_4m'].min())
            chair_normalized = (df['raise_chair_time'] - df['raise_chair_time'].min()) / (df['raise_chair_time'].max() - df['raise_chair_time'].min())
            df['frailty_physical_score'] = (gait_normalized + chair_normalized) / 2
            print(f"   ✅ Criada: frailty_physical_score")
        
        # Feature 3: Grupo etário
        # Categorizar idade em grupos para capturar efeitos não-lineares
        df['age_group'] = pd.cut(df['age'], bins=[0, 74, 79, 100], labels=['70-74', '75-79', '80+'])
        print(f"   ✅ Criada: age_group")
        
        # Feature 4: Índice de independência funcional combinado
        # Combina Katz (atividades básicas) e IADL (atividades instrumentais)
        if 'katz_index' in df.columns and 'iadl_grade' in df.columns:
            # Normalizar ambos para 0-1
            katz_norm = df['katz_index'] / df['katz_index'].max()
            iadl_norm = df['iadl_grade'] / df['iadl_grade'].max()
            df['functional_independence_score'] = (katz_norm + iadl_norm) / 2
            print(f"   ✅ Criada: functional_independence_score")
        
        # Feature 5: Risco cognitivo-psicológico
        # Combina cognição e depressão (fatores de risco importantes)
        if 'mmse_total_score' in df.columns and 'depression_total_score' in df.columns:
            # MMSE: quanto maior, melhor (inverter)
            # Depression: quanto maior, pior
            mmse_risk = 1 - (df['mmse_total_score'] / df['mmse_total_score'].max())
            depression_risk = df['depression_total_score'] / df['depression_total_score'].max()
            df['cognitive_psych_risk'] = (mmse_risk + depression_risk) / 2
            print(f"   ✅ Criada: cognitive_psych_risk")
        
        print(f"   📊 Total de novas features criadas: 5")
        
        # ====================================================================
        # 5. ENCODING DE VARIÁVEIS CATEGÓRICAS
        # ====================================================================
        print("\n5️⃣ Codificando variáveis categóricas...")
        
        # Atualizar lista de features categóricas
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Label Encoding para variáveis categóricas
        # (One-Hot Encoding seria melhor, mas com dataset pequeno, evitamos criar muitas features)
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le  # Salvar para uso futuro
            print(f"   ✅ Encoded: {col} ({len(le.classes_)} categorias)")
        
        # ====================================================================
        # 6. NORMALIZAÇÃO DE FEATURES NUMÉRICAS
        # ====================================================================
        print("\n6️⃣ Normalizando features numéricas...")
        
        # Identificar features numéricas após feature engineering
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Aplicar StandardScaler (média=0, desvio=1)
        # Importante para modelos como Logistic Regression e SVM
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        print(f"   ✅ {len(numeric_features)} features normalizadas (StandardScaler)")
        
        # ====================================================================
        # 7. FINALIZAÇÃO
        # ====================================================================
        self.feature_names = df.columns.tolist()
        
        print("\n" + "=" * 70)
        print("✅ PREPARAÇÃO CONCLUÍDA")
        print("=" * 70)
        print(f"📊 Shape final: {df.shape}")
        print(f"📋 Total de features: {len(self.feature_names)}")
        print(f"🎯 Targets: hospitalization_one_year, hospitalization_three_years")
        
        return df.values, y_1year.values, y_3years.values
    
    
    def split_data(self, X, y_1year, y_3years, test_size=0.3):
        """
        Divide os dados em conjuntos de treino e teste.
        
        Usa stratified split para manter a proporção de classes em ambos os conjuntos.
        Isso é especialmente importante para classes desbalanceadas.
        
        Args:
            X (np.array): Features preparadas
            y_1year (np.array): Target de 1 ano
            y_3years (np.array): Target de 3 anos
            test_size (float): Proporção do conjunto de teste (padrão: 30%)
        """
        print("\n" + "=" * 70)
        print("✂️ DIVISÃO DOS DADOS")
        print("=" * 70)
        
        # Dividir dados para target de 1 ano
        # stratify=y_1year garante que a proporção de classes seja mantida
        self.X_train, self.X_test, self.y_train_1y, self.y_test_1y = train_test_split(
            X, y_1year,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_1year  # Mantém proporção de classes
        )
        
        # Para 3 anos, usar os mesmos índices de divisão
        # Isso garante que os mesmos pacientes estejam em treino/teste
        _, _, self.y_train_3y, self.y_test_3y = train_test_split(
            X, y_3years,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_3years
        )
        
        print(f"📊 Conjunto de TREINO: {self.X_train.shape[0]} amostras ({(1-test_size)*100:.0f}%)")
        print(f"📊 Conjunto de TESTE:  {self.X_test.shape[0]} amostras ({test_size*100:.0f}%)")
        
        print(f"\n🎯 Distribuição - Hospitalização 1 ano:")
        print(f"   Treino - Sim: {self.y_train_1y.sum()} ({self.y_train_1y.mean()*100:.1f}%) | Não: {(self.y_train_1y==0).sum()} ({(1-self.y_train_1y.mean())*100:.1f}%)")
        print(f"   Teste  - Sim: {self.y_test_1y.sum()} ({self.y_test_1y.mean()*100:.1f}%) | Não: {(self.y_test_1y==0).sum()} ({(1-self.y_test_1y.mean())*100:.1f}%)")
        
        print(f"\n🎯 Distribuição - Hospitalização 3 anos:")
        print(f"   Treino - Sim: {self.y_train_3y.sum()} ({self.y_train_3y.mean()*100:.1f}%) | Não: {(self.y_train_3y==0).sum()} ({(1-self.y_train_3y.mean())*100:.1f}%)")
        print(f"   Teste  - Sim: {self.y_test_3y.sum()} ({self.y_test_3y.mean()*100:.1f}%) | Não: {(self.y_test_3y==0).sum()} ({(1-self.y_test_3y.mean())*100:.1f}%)")
        
        print("\n✅ Divisão concluída com sucesso!")
    
    
    def train_models(self, target='1year'):
        """
        Treina múltiplos modelos de classificação.
        
        Treina e compara diferentes algoritmos:
        1. Logistic Regression - Baseline simples e interpretável
        2. Decision Tree - Modelo não-linear simples
        3. Random Forest - Ensemble robusto
        4. Gradient Boosting - Modelo avançado
        
        Args:
            target (str): '1year' ou '3years' - qual target treinar
        """
        print("\n" + "=" * 70)
        print(f"🤖 TREINAMENTO DE MODELOS - {target.upper()}")
        print("=" * 70)
        
        # Selecionar target apropriado
        if target == '1year':
            y_train = self.y_train_1y
            y_test = self.y_test_1y
            models_dict = self.models_1y
        else:
            y_train = self.y_train_3y
            y_test = self.y_test_3y
            models_dict = self.models_3y
        
        # Calcular class weights para lidar com desbalanceamento
        # Dá mais peso à classe minoritária
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"⚖️ Class weights calculados: {class_weight_dict}")
        print(f"   (Dá mais peso à classe minoritária para balancear o aprendizado)\n")
        
        # ====================================================================
        # MODELO 1: LOGISTIC REGRESSION
        # ====================================================================
        print("1️⃣ Treinando Logistic Regression...")
        print("   📝 Modelo linear simples, altamente interpretável")
        print("   📝 Bom baseline para problemas de classificação")
        
        lr = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,              # Máximo de iterações
            class_weight='balanced',     # Balancear classes automaticamente
            C=0.1                        # Regularização forte (evita overfitting)
        )
        lr.fit(self.X_train, y_train)
        models_dict['Logistic Regression'] = lr
        print("   ✅ Treinado com sucesso!\n")
        
        # ====================================================================
        # MODELO 2: DECISION TREE
        # ====================================================================
        print("2️⃣ Treinando Decision Tree...")
        print("   📝 Modelo não-linear baseado em regras")
        print("   📝 Fácil de interpretar e visualizar")
        
        dt = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=5,                 # Limitar profundidade (evita overfitting)
            min_samples_split=10,        # Mínimo de amostras para dividir nó
            min_samples_leaf=5,          # Mínimo de amostras por folha
            class_weight='balanced'      # Balancear classes
        )
        dt.fit(self.X_train, y_train)
        models_dict['Decision Tree'] = dt
        print("   ✅ Treinado com sucesso!\n")
        
        # ====================================================================
        # MODELO 3: RANDOM FOREST
        # ====================================================================
        print("3️⃣ Treinando Random Forest...")
        print("   📝 Ensemble de múltiplas árvores de decisão")
        print("   📝 Robusto e geralmente com boa performance")
        
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,            # Número de árvores
            max_depth=5,                 # Profundidade máxima de cada árvore
            min_samples_split=10,        # Mínimo para dividir
            min_samples_leaf=5,          # Mínimo por folha
            class_weight='balanced',     # Balancear classes
            n_jobs=-1                    # Usar todos os cores disponíveis
        )
        rf.fit(self.X_train, y_train)
        models_dict['Random Forest'] = rf
        print("   ✅ Treinado com sucesso!\n")
        
        # ====================================================================
        # MODELO 4: GRADIENT BOOSTING
        # ====================================================================
        print("4️⃣ Treinando Gradient Boosting...")
        print("   📝 Ensemble sequencial que corrige erros iterativamente")
        print("   📝 Geralmente alta performance, mas risco de overfitting")
        
        gb = GradientBoostingClassifier(
            random_state=self.random_state,
            n_estimators=100,            # Número de boosting stages
            learning_rate=0.05,          # Taxa de aprendizado baixa (mais conservador)
            max_depth=3,                 # Árvores rasas (evita overfitting)
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8                # Usar 80% dos dados em cada iteração
        )
        gb.fit(self.X_train, y_train)
        models_dict['Gradient Boosting'] = gb
        print("   ✅ Treinado com sucesso!\n")
        
        print("=" * 70)
        print(f"✅ TODOS OS MODELOS TREINADOS - {target.upper()}")
        print(f"📊 Total de modelos: {len(models_dict)}")
        print("=" * 70)
    
    
    def evaluate_models(self, target='1year'):
        """
        Avalia todos os modelos treinados.
        
        Calcula múltiplas métricas para cada modelo:
        - Accuracy: Proporção de acertos
        - Precision: Dos preditos como positivo, quantos são realmente positivos
        - Recall: Dos realmente positivos, quantos conseguimos identificar
        - F1-Score: Média harmônica de precision e recall
        - ROC-AUC: Capacidade de discriminação do modelo
        
        Args:
            target (str): '1year' ou '3years'
            
        Returns:
            pd.DataFrame: Tabela com métricas de todos os modelos
        """
        print("\n" + "=" * 70)
        print(f"📊 AVALIAÇÃO DE MODELOS - {target.upper()}")
        print("=" * 70)
        
        # Selecionar dados apropriados
        if target == '1year':
            y_train = self.y_train_1y
            y_test = self.y_test_1y
            models_dict = self.models_1y
        else:
            y_train = self.y_train_3y
            y_test = self.y_test_3y
            models_dict = self.models_3y
        
        # Dicionário para armazenar resultados
        results = []
        
        print("\n🔍 Avaliando cada modelo...\n")
        
        for model_name, model in models_dict.items():
            print(f"📈 {model_name}")
            print("-" * 70)
            
            # Predições no conjunto de treino
            y_train_pred = model.predict(self.X_train)
            y_train_proba = model.predict_proba(self.X_train)
            
            # Predições no conjunto de teste
            y_test_pred = model.predict(self.X_test)
            y_test_proba = model.predict_proba(self.X_test)
            
            # Determinar se é classificação binária ou multiclasse
            n_classes = len(np.unique(y_train))
            is_binary = n_classes == 2
            
            # Calcular métricas para TREINO
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            if is_binary:
                # Classificação binária - usar métricas padrão
                train_precision = precision_score(y_train, y_train_pred, zero_division=0)
                train_recall = recall_score(y_train, y_train_pred, zero_division=0)
                train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
                train_roc_auc = roc_auc_score(y_train, y_train_proba[:, 1])
            else:
                # Classificação multiclasse - usar average='weighted'
                train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_roc_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='weighted')
            
            # Calcular métricas para TESTE
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            if is_binary:
                # Classificação binária
                test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
                test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
            else:
                # Classificação multiclasse
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
            
            # Exibir métricas
            print(f"   TREINO  → Acc: {train_accuracy:.3f} | Prec: {train_precision:.3f} | Rec: {train_recall:.3f} | F1: {train_f1:.3f} | AUC: {train_roc_auc:.3f}")
            print(f"   TESTE   → Acc: {test_accuracy:.3f} | Prec: {test_precision:.3f} | Rec: {test_recall:.3f} | F1: {test_f1:.3f} | AUC: {test_roc_auc:.3f}")
            
            # Verificar overfitting
            overfit_gap = train_roc_auc - test_roc_auc
            if overfit_gap > 0.15:
                print(f"   ⚠️ ALERTA: Possível overfitting detectado (gap AUC: {overfit_gap:.3f})")
            elif overfit_gap > 0.10:
                print(f"   ⚡ Leve overfitting (gap AUC: {overfit_gap:.3f})")
            else:
                print(f"   ✅ Boa generalização (gap AUC: {overfit_gap:.3f})")
            
            print()
            
            # Armazenar resultados
            results.append({
                'Modelo': model_name,
                'Train_Accuracy': train_accuracy,
                'Test_Accuracy': test_accuracy,
                'Train_Precision': train_precision,
                'Test_Precision': test_precision,
                'Train_Recall': train_recall,
                'Test_Recall': test_recall,
                'Train_F1': train_f1,
                'Test_F1': test_f1,
                'Train_ROC_AUC': train_roc_auc,
                'Test_ROC_AUC': test_roc_auc,
                'Overfit_Gap': overfit_gap
            })
        
        # Criar DataFrame com resultados
        results_df = pd.DataFrame(results)
        
        # Identificar melhor modelo (baseado em Test ROC-AUC)
        best_model_idx = results_df['Test_ROC_AUC'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'Modelo']
        best_auc = results_df.loc[best_model_idx, 'Test_ROC_AUC']
        
        print("=" * 70)
        print(f"🏆 MELHOR MODELO: {best_model_name}")
        print(f"   ROC-AUC no teste: {best_auc:.3f}")
        print("=" * 70)
        
        # Salvar resultados
        results_df.to_csv(f'outputs/model_comparison_{target}.csv', index=False)
        print(f"\n💾 Resultados salvos em: outputs/model_comparison_{target}.csv")
        
        return results_df
    
    
    def plot_confusion_matrices(self, target='1year'):
        """
        Plota matrizes de confusão para todos os modelos.
        
        A matriz de confusão mostra:
        - True Positives (TP): Corretamente predito como hospitalizado
        - True Negatives (TN): Corretamente predito como não hospitalizado
        - False Positives (FP): Incorretamente predito como hospitalizado
        - False Negatives (FN): Incorretamente predito como não hospitalizado
        
        Args:
            target (str): '1year' ou '3years'
        """
        print(f"\n📊 Gerando matrizes de confusão - {target}...")
        
        # Selecionar dados apropriados
        if target == '1year':
            y_test = self.y_test_1y
            models_dict = self.models_1y
        else:
            y_test = self.y_test_3y
            models_dict = self.models_3y
        
        # Criar subplots
        n_models = len(models_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            # Predições
            y_pred = model.predict(self.X_test)
            
            # Calcular matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            
            # Plotar
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, square=True,
                       xticklabels=['Não Hosp.', 'Hosp.'],
                       yticklabels=['Não Hosp.', 'Hosp.'])
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Real', fontsize=10)
            axes[idx].set_xlabel('Predito', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'outputs/confusion_matrix_{target}.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: outputs/confusion_matrix_{target}.png")
        plt.close()
    
    
    def plot_roc_curves(self, target='1year'):
        """
        Plota curvas ROC para todos os modelos.
        
        A curva ROC (Receiver Operating Characteristic) mostra o trade-off entre
        True Positive Rate (Recall) e False Positive Rate em diferentes thresholds.
        
        AUC (Area Under Curve):
        - 1.0: Classificador perfeito
        - 0.5: Classificador aleatório
        - < 0.5: Pior que aleatório
        
        Args:
            target (str): '1year' ou '3years'
        """
        print(f"\n📈 Gerando curvas ROC - {target}...")
        
        # Selecionar dados apropriados
        if target == '1year':
            y_test = self.y_test_1y
            models_dict = self.models_1y
        else:
            y_test = self.y_test_3y
            models_dict = self.models_3y
        
        plt.figure(figsize=(10, 8))
        
        # Determinar se é classificação binária ou multiclasse
        n_classes = len(np.unique(y_test))
        is_binary = n_classes == 2
        
        # Plotar curva ROC para cada modelo
        for model_name, model in models_dict.items():
            # Obter probabilidades
            y_proba = model.predict_proba(self.X_test)
            
            if is_binary:
                # Classificação binária - usar apenas probabilidade da classe positiva
                y_proba_pos = y_proba[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
                roc_auc = auc(fpr, tpr)
            else:
                # Classificação multiclasse - calcular ROC-AUC médio
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                
                # Calcular ROC-AUC para cada classe e fazer média
                roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
                
                # Para visualização, usar micro-average
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
            
            # Plotar
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Linha de referência (classificador aleatório)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'Curvas ROC - Hospitalização {target}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'outputs/roc_curve_{target}.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: outputs/roc_curve_{target}.png")
        plt.close()
    
    
    def plot_feature_importance(self, target='1year', top_n=15):
        """
        Plota importância das features para modelos baseados em árvores.
        
        Feature importance indica quais variáveis têm maior influência nas predições.
        Isso ajuda a entender quais fatores são mais relevantes para hospitalização.
        
        Args:
            target (str): '1year' ou '3years'
            top_n (int): Número de features mais importantes a exibir
        """
        print(f"\n🔍 Gerando gráficos de feature importance - {target}...")
        
        # Selecionar modelos apropriados
        if target == '1year':
            models_dict = self.models_1y
        else:
            models_dict = self.models_3y
        
        # Filtrar apenas modelos com feature_importances_
        tree_models = {name: model for name, model in models_dict.items() 
                      if hasattr(model, 'feature_importances_')}
        
        if not tree_models:
            print("   ⚠️ Nenhum modelo com feature importance disponível")
            return
        
        # Criar subplots
        n_models = len(tree_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(tree_models.items()):
            # Obter importâncias
            importances = model.feature_importances_
            
            # Criar DataFrame
            feature_imp_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plotar
            axes[idx].barh(range(len(feature_imp_df)), feature_imp_df['importance'], color='steelblue')
            axes[idx].set_yticks(range(len(feature_imp_df)))
            axes[idx].set_yticklabels(feature_imp_df['feature'], fontsize=9)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importância', fontsize=10)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'outputs/feature_importance_{target}.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Salvo: outputs/feature_importance_{target}.png")
        plt.close()
    
    
    def save_models(self):
        """
        Salva os modelos treinados em disco.
        
        Os modelos são salvos usando pickle para uso futuro.
        """
        print("\n💾 Salvando modelos...")
        
        # Salvar modelos de 1 ano
        for model_name, model in self.models_1y.items():
            filename = f"models/{model_name.replace(' ', '_').lower()}_1year.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ {filename}")
        
        # Salvar modelos de 3 anos
        for model_name, model in self.models_3y.items():
            filename = f"models/{model_name.replace(' ', '_').lower()}_3years.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ {filename}")
        
        # Salvar scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✅ models/scaler.pkl")
        
        print("\n✅ Todos os modelos salvos com sucesso!")
    
    
    def generate_analysis_report(self):
        """
        Gera um relatório completo com análise dos resultados.
        
        Este método:
        1. Analisa os resultados de ambos os targets
        2. Identifica os melhores modelos
        3. Explica as visualizações geradas
        4. Fornece conclusões e recomendações
        """
        print("\n" + "=" * 70)
        print("📊 GERANDO RELATÓRIO DE ANÁLISE")
        print("=" * 70)
        
        # Carregar resultados
        try:
            results_1y = pd.read_csv('outputs/model_comparison_1year.csv')
            results_3y = pd.read_csv('outputs/model_comparison_3years.csv')
        except:
            print("⚠️ Não foi possível carregar os resultados para análise.")
            return
        
        # Criar relatório em markdown
        report_lines = []
        
        # Cabeçalho
        report_lines.append("# Relatório de Análise: Modelo Preditivo de Hospitalização\n")
        report_lines.append(f"**Data de Geração**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        report_lines.append("**Autor**: Rafael Zanarino\n")
        report_lines.append("---\n\n")
        
        # Sumário Executivo
        report_lines.append("## 📋 Sumário Executivo\n\n")
        report_lines.append("Este relatório apresenta os resultados da modelagem preditiva de hospitalização ")
        report_lines.append("de pacientes idosos em dois horizontes temporais: **1 ano** e **3 anos**.\n\n")
        
        # Identificar melhores modelos
        best_1y_idx = results_1y['Test_ROC_AUC'].idxmax()
        best_1y_name = results_1y.loc[best_1y_idx, 'Modelo']
        best_1y_auc = results_1y.loc[best_1y_idx, 'Test_ROC_AUC']
        
        best_3y_idx = results_3y['Test_ROC_AUC'].idxmax()
        best_3y_name = results_3y.loc[best_3y_idx, 'Modelo']
        best_3y_auc = results_3y.loc[best_3y_idx, 'Test_ROC_AUC']
        
        report_lines.append("### 🏆 Melhores Modelos\n\n")
        report_lines.append(f"- **Hospitalização 1 ano**: {best_1y_name} (ROC-AUC: {best_1y_auc:.3f})\n")
        report_lines.append(f"- **Hospitalização 3 anos**: {best_3y_name} (ROC-AUC: {best_3y_auc:.3f})\n\n")
        
        # Análise detalhada - 1 ano
        report_lines.append("---\n\n")
        report_lines.append("## 🎯 Análise: Predição de Hospitalização em 1 Ano\n\n")
        
        report_lines.append("### Comparação de Modelos\n\n")
        report_lines.append("| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overfitting |\n")
        report_lines.append("|--------|----------|-----------|--------|----------|---------|-------------|\n")
        
        for _, row in results_1y.iterrows():
            overfit_status = "✅" if row['Overfit_Gap'] < 0.10 else ("⚡" if row['Overfit_Gap'] < 0.15 else "⚠️")
            report_lines.append(f"| {row['Modelo']} | {row['Test_Accuracy']:.3f} | {row['Test_Precision']:.3f} | ")
            report_lines.append(f"{row['Test_Recall']:.3f} | {row['Test_F1']:.3f} | {row['Test_ROC_AUC']:.3f} | ")
            report_lines.append(f"{overfit_status} {row['Overfit_Gap']:.3f} |\n")
        
        report_lines.append("\n")
        
        # Interpretação do melhor modelo - 1 ano
        report_lines.append(f"### 🔍 Análise do Melhor Modelo: {best_1y_name}\n\n")
        
        best_1y_row = results_1y.loc[best_1y_idx]
        
        report_lines.append(f"**Por que este modelo foi escolhido?**\n\n")
        report_lines.append(f"O **{best_1y_name}** apresentou o melhor desempenho com ROC-AUC de **{best_1y_auc:.3f}**, ")
        report_lines.append(f"indicando {'excelente' if best_1y_auc > 0.9 else ('muito boa' if best_1y_auc > 0.8 else ('boa' if best_1y_auc > 0.7 else 'razoável'))} ")
        report_lines.append(f"capacidade de discriminação entre pacientes que serão e não serão hospitalizados.\n\n")
        
        report_lines.append(f"**Métricas de Performance:**\n\n")
        report_lines.append(f"- **Accuracy**: {best_1y_row['Test_Accuracy']:.1%} - Proporção de predições corretas\n")
        report_lines.append(f"- **Precision**: {best_1y_row['Test_Precision']:.1%} - Dos preditos como 'alto risco', {best_1y_row['Test_Precision']:.1%} realmente foram hospitalizados\n")
        report_lines.append(f"- **Recall**: {best_1y_row['Test_Recall']:.1%} - Dos pacientes hospitalizados, {best_1y_row['Test_Recall']:.1%} foram corretamente identificados\n")
        report_lines.append(f"- **F1-Score**: {best_1y_row['Test_F1']:.3f} - Balanço entre precision e recall\n\n")
        
        # Análise de overfitting
        if best_1y_row['Overfit_Gap'] < 0.10:
            report_lines.append(f"✅ **Generalização Excelente**: Gap de {best_1y_row['Overfit_Gap']:.3f} indica que o modelo generaliza bem para novos dados.\n\n")
        elif best_1y_row['Overfit_Gap'] < 0.15:
            report_lines.append(f"⚡ **Leve Overfitting**: Gap de {best_1y_row['Overfit_Gap']:.3f} sugere leve memorização dos dados de treino, mas ainda aceitável.\n\n")
        else:
            report_lines.append(f"⚠️ **Overfitting Detectado**: Gap de {best_1y_row['Overfit_Gap']:.3f} indica que o modelo pode estar memorizando os dados de treino.\n\n")
        
        # Análise detalhada - 3 anos
        report_lines.append("---\n\n")
        report_lines.append("## 🎯 Análise: Predição de Hospitalização em 3 Anos\n\n")
        
        report_lines.append("### Comparação de Modelos\n\n")
        report_lines.append("| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overfitting |\n")
        report_lines.append("|--------|----------|-----------|--------|----------|---------|-------------|\n")
        
        for _, row in results_3y.iterrows():
            overfit_status = "✅" if row['Overfit_Gap'] < 0.10 else ("⚡" if row['Overfit_Gap'] < 0.15 else "⚠️")
            report_lines.append(f"| {row['Modelo']} | {row['Test_Accuracy']:.3f} | {row['Test_Precision']:.3f} | ")
            report_lines.append(f"{row['Test_Recall']:.3f} | {row['Test_F1']:.3f} | {row['Test_ROC_AUC']:.3f} | ")
            report_lines.append(f"{overfit_status} {row['Overfit_Gap']:.3f} |\n")
        
        report_lines.append("\n")
        
        # Interpretação do melhor modelo - 3 anos
        report_lines.append(f"### 🔍 Análise do Melhor Modelo: {best_3y_name}\n\n")
        
        best_3y_row = results_3y.loc[best_3y_idx]
        
        report_lines.append(f"**Por que este modelo foi escolhido?**\n\n")
        report_lines.append(f"O **{best_3y_name}** apresentou o melhor desempenho com ROC-AUC de **{best_3y_auc:.3f}**, ")
        report_lines.append(f"indicando {'excelente' if best_3y_auc > 0.9 else ('muito boa' if best_3y_auc > 0.8 else ('boa' if best_3y_auc > 0.7 else 'razoável'))} ")
        report_lines.append(f"capacidade de discriminação.\n\n")
        
        report_lines.append(f"**Métricas de Performance:**\n\n")
        report_lines.append(f"- **Accuracy**: {best_3y_row['Test_Accuracy']:.1%}\n")
        report_lines.append(f"- **Precision**: {best_3y_row['Test_Precision']:.1%}\n")
        report_lines.append(f"- **Recall**: {best_3y_row['Test_Recall']:.1%}\n")
        report_lines.append(f"- **F1-Score**: {best_3y_row['Test_F1']:.3f}\n\n")
        
        # Explicação das Visualizações
        report_lines.append("---\n\n")
        report_lines.append("## 📊 Explicação das Visualizações\n\n")
        
        # Confusion Matrix
        report_lines.append("### 1. Matrizes de Confusão\n\n")
        report_lines.append("**Arquivos**: `confusion_matrix_1year.png` e `confusion_matrix_3years.png`\n\n")
        report_lines.append("**O que mostram:**\n\n")
        report_lines.append("As matrizes de confusão visualizam os acertos e erros de cada modelo:\n\n")
        report_lines.append("```\n")
        report_lines.append("                Predito\n")
        report_lines.append("             Não Hosp.  Hosp.\n")
        report_lines.append("Real  Não H.    TN       FP     ← Falsos Alarmes\n")
        report_lines.append("      Hosp.     FN       TP     ← Casos Perdidos\n")
        report_lines.append("```\n\n")
        report_lines.append("- **TN (True Negative)**: Pacientes corretamente identificados como baixo risco\n")
        report_lines.append("- **TP (True Positive)**: Pacientes corretamente identificados como alto risco\n")
        report_lines.append("- **FP (False Positive)**: Falsos alarmes - preditos como alto risco mas não hospitalizados\n")
        report_lines.append("- **FN (False Negative)**: Casos perdidos - não identificados mas foram hospitalizados\n\n")
        report_lines.append("**Como interpretar**: Quanto maior os valores na diagonal (TN e TP), melhor o modelo.\n\n")
        
        # ROC Curves
        report_lines.append("### 2. Curvas ROC\n\n")
        report_lines.append("**Arquivos**: `roc_curve_1year.png` e `roc_curve_3years.png`\n\n")
        report_lines.append("**O que mostram:**\n\n")
        report_lines.append("As curvas ROC (Receiver Operating Characteristic) mostram o trade-off entre:\n")
        report_lines.append("- **True Positive Rate (Recall)**: Taxa de acerto nos casos positivos\n")
        report_lines.append("- **False Positive Rate**: Taxa de falsos alarmes\n\n")
        report_lines.append("**Interpretação da AUC (Area Under Curve)**:\n")
        report_lines.append("- **0.9 - 1.0**: Excelente discriminação\n")
        report_lines.append("- **0.8 - 0.9**: Muito boa discriminação\n")
        report_lines.append("- **0.7 - 0.8**: Boa discriminação\n")
        report_lines.append("- **0.6 - 0.7**: Razoável\n")
        report_lines.append("- **0.5**: Aleatório (jogar moeda)\n\n")
        report_lines.append("**Como interpretar**: Quanto mais próxima a curva do canto superior esquerdo, melhor o modelo.\n\n")
        
        # Feature Importance
        report_lines.append("### 3. Importância das Features\n\n")
        report_lines.append("**Arquivos**: `feature_importance_1year.png` e `feature_importance_3years.png`\n\n")
        report_lines.append("**O que mostram:**\n\n")
        report_lines.append("Estes gráficos mostram quais variáveis têm maior influência nas predições dos modelos baseados em árvores ")
        report_lines.append("(Decision Tree, Random Forest, Gradient Boosting).\n\n")
        report_lines.append("**Como interpretar**:\n")
        report_lines.append("- Features no topo da lista têm maior impacto nas predições\n")
        report_lines.append("- Ajuda a entender quais fatores clínicos são mais relevantes\n")
        report_lines.append("- Útil para validação clínica (as features importantes fazem sentido médico?)\n\n")
        
        # Conclusões e Recomendações
        report_lines.append("---\n\n")
        report_lines.append("## 💡 Conclusões e Recomendações\n\n")
        
        report_lines.append("### Principais Achados\n\n")
        report_lines.append(f"1. **Modelo mais eficaz para 1 ano**: {best_1y_name} com AUC de {best_1y_auc:.3f}\n")
        report_lines.append(f"2. **Modelo mais eficaz para 3 anos**: {best_3y_name} com AUC de {best_3y_auc:.3f}\n")
        
        # Comparar performance entre 1 e 3 anos
        if best_1y_auc > best_3y_auc:
            report_lines.append(f"3. **Predição de curto prazo** (1 ano) apresentou melhor performance que longo prazo (3 anos)\n")
        else:
            report_lines.append(f"3. **Predição de longo prazo** (3 anos) apresentou melhor performance que curto prazo (1 ano)\n")
        
        report_lines.append("\n### Recomendações de Uso\n\n")
        
        report_lines.append("**Para uso clínico:**\n\n")
        report_lines.append(f"1. Utilizar o **{best_1y_name}** para identificar pacientes em risco de hospitalização no próximo ano\n")
        report_lines.append(f"2. Utilizar o **{best_3y_name}** para planejamento de cuidados de longo prazo\n")
        report_lines.append("3. Considerar intervenções preventivas para pacientes identificados como alto risco\n")
        report_lines.append("4. Monitorar continuamente a performance dos modelos com novos dados\n\n")
        
        report_lines.append("### Limitações\n\n")
        report_lines.append("⚠️ **Importante considerar:**\n\n")
        report_lines.append("1. **Dataset pequeno** (117 observações) - limita a confiabilidade estatística\n")
        report_lines.append("2. **Validação externa necessária** - testar em nova população antes de uso clínico\n")
        report_lines.append("3. **Modelos não substituem julgamento clínico** - usar como ferramenta de apoio à decisão\n")
        report_lines.append("4. **Re-treinamento periódico** - atualizar modelos com novos dados regularmente\n\n")
        
        report_lines.append("### Próximos Passos\n\n")
        report_lines.append("1. ✅ Coletar mais dados para aumentar robustez\n")
        report_lines.append("2. ✅ Validar em população externa\n")
        report_lines.append("3. ✅ Desenvolver interface de uso clínico\n")
        report_lines.append("4. ✅ Implementar monitoramento contínuo de performance\n")
        report_lines.append("5. ✅ Realizar estudos de impacto clínico\n\n")
        
        report_lines.append("---\n\n")
        report_lines.append("**Relatório gerado automaticamente pelo HospitalizationPredictor**\n")
        
        # Salvar relatório
        report_content = ''.join(report_lines)
        with open('outputs/RELATORIO_ANALISE.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("✅ Relatório de análise gerado!")
        print("   📄 outputs/RELATORIO_ANALISE.md")
        
        # Também exibir resumo no console
        print("\n" + "=" * 70)
        print("📋 RESUMO DA ANÁLISE")
        print("=" * 70)
        print(f"\n🏆 MELHOR MODELO - 1 ANO: {best_1y_name}")
        print(f"   ROC-AUC: {best_1y_auc:.3f}")
        print(f"   Accuracy: {best_1y_row['Test_Accuracy']:.1%}")
        print(f"   F1-Score: {best_1y_row['Test_F1']:.3f}")
        
        print(f"\n🏆 MELHOR MODELO - 3 ANOS: {best_3y_name}")
        print(f"   ROC-AUC: {best_3y_auc:.3f}")
        print(f"   Accuracy: {best_3y_row['Test_Accuracy']:.1%}")
        print(f"   F1-Score: {best_3y_row['Test_F1']:.3f}")
        
        print("\n" + "=" * 70)
        print("📊 VISUALIZAÇÕES GERADAS")
        print("=" * 70)
        print("\n1. Matrizes de Confusão:")
        print("   - outputs/confusion_matrix_1year.png")
        print("   - outputs/confusion_matrix_3years.png")
        print("   → Mostram acertos (diagonal) e erros de cada modelo")
        
        print("\n2. Curvas ROC:")
        print("   - outputs/roc_curve_1year.png")
        print("   - outputs/roc_curve_3years.png")
        print("   → Mostram capacidade de discriminação (quanto maior AUC, melhor)")
        
        print("\n3. Importância das Features:")
        print("   - outputs/feature_importance_1year.png")
        print("   - outputs/feature_importance_3years.png")
        print("   → Mostram quais variáveis mais influenciam as predições")
        
        print("\n" + "=" * 70)
        print("✅ ANÁLISE COMPLETA!")
        print("=" * 70)
        print("\n📄 Leia o relatório completo em: outputs/RELATORIO_ANALISE.md")
        print("=" * 70)



# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal que executa todo o pipeline.
    """
    print("\n" + "=" * 70)
    print("🏥 MODELO PREDITIVO DE HOSPITALIZAÇÃO DE PACIENTES IDOSOS")
    print("=" * 70)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Inicializar preditor
    predictor = HospitalizationPredictor(random_state=42)
    
    # 1. Carregar dados
    predictor.load_data()
    
    # 2. Preparar features
    X, y_1year, y_3years = predictor.prepare_features()
    
    # 3. Dividir dados
    predictor.split_data(X, y_1year, y_3years, test_size=0.3)
    
    # 4. Treinar modelos para 1 ano
    predictor.train_models(target='1year')
    
    # 5. Treinar modelos para 3 anos
    predictor.train_models(target='3years')
    
    # 6. Avaliar modelos
    results_1y = predictor.evaluate_models(target='1year')
    results_3y = predictor.evaluate_models(target='3years')
    
    # 7. Gerar visualizações
    predictor.plot_confusion_matrices(target='1year')
    predictor.plot_confusion_matrices(target='3years')
    predictor.plot_roc_curves(target='1year')
    predictor.plot_roc_curves(target='3years')
    predictor.plot_feature_importance(target='1year')
    predictor.plot_feature_importance(target='3years')
    
    # 8. Salvar modelos
    predictor.save_models()
    
    # 9. Gerar relatório de análise
    predictor.generate_analysis_report()
    
    print(f"\n⏰ Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✨ Obrigado por usar o HospitalizationPredictor! ✨\n")


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    main()
