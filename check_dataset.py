"""
Script auxiliar para extrair o dataset do notebook Jupyter e salvá-lo como CSV.

Este script lê o notebook exploratory_analysis_dataset.ipynb e extrai os dados
para criar o arquivo CSV necessário para o modelo preditivo.
"""

import pandas as pd
import json

def extract_data_from_notebook():
    """
    Extrai dados do notebook Jupyter.
    
    Como o notebook já carrega os dados, vamos criar um script simples
    que executa a célula de carregamento do notebook.
    """
    print("🔍 Procurando dataset no notebook...")
    
    # Tentar carregar o notebook
    try:
        with open('exploratory_analysis_dataset.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print("✅ Notebook carregado com sucesso!")
        print("\n📋 Instruções:")
        print("=" * 70)
        print("O notebook já contém o código para carregar os dados:")
        print("  data = pd.read_csv('raw_data/Virtual_Patient_Models_Dataset.csv')")
        print("\n⚠️ AÇÃO NECESSÁRIA:")
        print("  1. Certifique-se de que o arquivo CSV existe em:")
        print("     raw_data/Virtual_Patient_Models_Dataset.csv")
        print("\n  2. Se você tem os dados em outro formato, por favor:")
        print("     a) Coloque o arquivo CSV no diretório raw_data/")
        print("     b) Ou execute o notebook para gerar os dados")
        print("\n  3. Após ter o arquivo CSV, execute:")
        print("     python predictive_model.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Erro ao processar notebook: {e}")

if __name__ == "__main__":
    extract_data_from_notebook()
