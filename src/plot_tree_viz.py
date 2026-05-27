import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import os
import yaml

# ========== ЗАГРУЗКА ==========
with open('params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
    tree_params = params['tree_models']

# Загружаем признаки для подписей
X_train = pd.read_csv('data/processed/X_train.csv')
feature_names = X_train.columns.tolist()

os.makedirs('reports/tree/trees', exist_ok=True)

# ========== ВИЗУАЛИЗАЦИЯ ДЕРЕВЬЕВ ==========
for model_name in tree_params.keys():
    if not tree_params[model_name].get('enabled', True):
        continue
    
    model_path = f'models/tree/{model_name}.pkl'
    try:
        model = joblib.load(model_path)
        
        # Полное дерево (уменьшенное для читаемости)
        plt.figure(figsize=(20, 10))
        plot_tree(model, 
                 feature_names=feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 max_depth=None)  # полное дерево
        plt.title(f'Полное дерево - {model_name}')
        plt.tight_layout()
        plt.savefig(f'reports/tree/trees/{model_name}_full.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Первые 3 уровня (для отчёта)
        plt.figure(figsize=(20, 8))
        plot_tree(model, 
                 feature_names=feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=12,
                 max_depth=3)  # только первые 3 уровня
        plt.title(f'Первые узлы дерева - {model_name} (max_depth=3)')
        plt.tight_layout()
        plt.savefig(f'reports/tree/trees/{model_name}_first_nodes.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Дерево {model_name} визуализировано")
        
    except FileNotFoundError:
        print(f"⚠️ Модель {model_name} не найдена")

print("\n✅ Визуализация деревьев завершена!")
print(f"📁 Файлы сохранены в: reports/tree/trees/")