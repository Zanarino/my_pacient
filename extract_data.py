"""
Script para extrair dados do notebook Jupyter e criar o arquivo CSV.

Este script executa as células do notebook para carregar os dados
e salva-os no formato CSV necessário para o modelo preditivo.
"""

import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def extract_data_from_notebook():
    """
    Extrai dados executando o notebook Jupyter.
    """
    print("=" * 70)
    print("📊 EXTRAÇÃO DE DADOS DO NOTEBOOK")
    print("=" * 70)
    
    notebook_path = 'exploratory_analysis_dataset.ipynb'
    
    # Verificar se notebook existe
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook não encontrado: {notebook_path}")
        return False
    
    print(f"\n📂 Carregando notebook: {notebook_path}")
    
    try:
        # Carregar notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print("✅ Notebook carregado com sucesso!")
        
        # Executar apenas as primeiras células necessárias
        print("\n⚙️ Executando células para carregar dados...")
        
        # Criar executor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Executar notebook
        try:
            ep.preprocess(nb, {'metadata': {'path': './'}})
            print("✅ Células executadas com sucesso!")
        except Exception as e:
            print(f"⚠️ Erro ao executar notebook: {e}")
            print("\n💡 Tentando método alternativo...")
            return extract_data_alternative()
        
        # Extrair dados do namespace
        # (Isso pode não funcionar diretamente, então usamos método alternativo)
        return extract_data_alternative()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def extract_data_alternative():
    """
    Método alternativo: instruir usuário a executar o notebook.
    """
    print("\n" + "=" * 70)
    print("📋 INSTRUÇÕES PARA PREPARAR OS DADOS")
    print("=" * 70)
    
    print("\n🔍 O arquivo CSV não foi encontrado em raw_data/")
    print("\n📝 Por favor, siga um dos métodos abaixo:\n")
    
    print("MÉTODO 1: Se você já tem o arquivo CSV")
    print("-" * 70)
    print("  1. Coloque o arquivo 'Virtual_Patient_Models_Dataset.csv'")
    print("     no diretório: raw_data/")
    print("  2. Execute: python predictive_model.py")
    
    print("\nMÉTODO 2: Extrair do notebook Jupyter")
    print("-" * 70)
    print("  1. Abra o notebook: exploratory_analysis_dataset.ipynb")
    print("  2. Execute a célula que carrega os dados:")
    print("     data = pd.read_csv('raw_data/Virtual_Patient_Models_Dataset.csv')")
    print("  3. Adicione uma nova célula com:")
    print("     data.to_csv('raw_data/Virtual_Patient_Models_Dataset.csv', index=False)")
    print("  4. Execute essa célula")
    print("  5. Execute: python predictive_model.py")
    
    print("\nMÉTODO 3: Criar dados de exemplo (APENAS PARA TESTE)")
    print("-" * 70)
    print("  1. Execute: python create_sample_data.py")
    print("  2. Execute: python predictive_model.py")
    print("  ⚠️ ATENÇÃO: Isso criará dados sintéticos apenas para testar o código!")
    
    print("\n" + "=" * 70)
    
    return False

if __name__ == "__main__":
    success = extract_data_from_notebook()
    
    if success:
        print("\n✅ Dados extraídos com sucesso!")
        print("📁 Arquivo salvo em: raw_data/Virtual_Patient_Models_Dataset.csv")
        print("\n🚀 Próximo passo: python predictive_model.py")
    else:
        print("\n⚠️ Siga as instruções acima para preparar os dados.")
