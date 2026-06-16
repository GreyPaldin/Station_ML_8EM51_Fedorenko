import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    max_error, median_absolute_error
)
from sklearn.preprocessing import StandardScaler
import yaml
from datetime import datetime
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import threading

# ========== ЗАГРУЗКА ПАРАМЕТРОВ ==========
with open('params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
    model_params = params['linear_models']

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

os.makedirs('reports/learning_curves', exist_ok=True)
os.makedirs('metrics/learning_curves', exist_ok=True)

# ========== 1. КРИВАЯ ОБУЧЕНИЯ (размер выборки → ошибка) ==========
def plot_learning_curve(model, model_name, X, y):
    """Кривая обучения: ошибка от размера выборки"""
    print(f"Строю learning curve для {model_name}...")
    
    train_sizes = np.linspace(0.1, 1.0, 10) * len(X)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        size = int(size)
        X_subset = X[:size]
        y_subset = y[:size]
        
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_subset, y_subset)
        
        train_pred = model_copy.predict(X_subset)
        val_pred = model_copy.predict(X_val)
        
        train_scores.append(mean_squared_error(y_subset, train_pred))
        val_scores.append(mean_squared_error(y_val, val_pred))
    
    # График
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Train MSE', color='blue', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation MSE', color='red', linewidth=2)
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('MSE')
    plt.title(f'Кривая обучения - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сохраняем
    plt.savefig(f'reports/learning_curves/{model_name}_learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сохраняем данные
    data = {
        'train_sizes': train_sizes.tolist(),
        'train_mse': train_scores,
        'val_mse': val_scores
    }
    with open(f'metrics/learning_curves/{model_name}_learning.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Сохранено")

# ========== 2. КРИВАЯ СХОДИМОСТИ (итерации → ошибка) ==========
def plot_convergence_curve(model, model_name, X, y, max_iter=1000):
    """Кривая сходимости: ошибка от числа итераций"""
    print(f"Строю convergence curve для {model_name}...")
    
    if not hasattr(model, 'max_iter'):
        print(f"   Модель {model_name} не поддерживает итерации")
        return
    
    train_errors = []
    val_errors = []
    iterations = np.linspace(10, max_iter, 20, dtype=int)
    
    for n_iter in iterations:
        model_copy = model.__class__(max_iter=n_iter, **{k:v for k,v in model.get_params().items() if k != 'max_iter'})
        model_copy.fit(X, y)
        
        train_pred = model_copy.predict(X)
        val_pred = model_copy.predict(X_val)
        
        train_errors.append(mean_squared_error(y, train_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))
    
    # График
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_errors, 'o-', label='Train MSE', color='blue', linewidth=2)
    plt.plot(iterations, val_errors, 'o-', label='Validation MSE', color='red', linewidth=2)
    plt.xlabel('Количество итераций')
    plt.ylabel('MSE')
    plt.title(f'Кривая сходимости - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'reports/learning_curves/{model_name}_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Сохранено")

# ========== 3. КРИВАЯ ОСТАТКОВ ==========
def plot_residuals(model, model_name, X_train, y_train, X_test, y_test):
    """Анализ остатков: предсказания vs ошибки"""
    print(f"Строю residuals plot для {model_name}...")
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_residuals = y_train - train_pred
    test_residuals = y_test - test_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Остатки vs Предсказания (train)
    axes[0, 0].scatter(train_pred, train_residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Предсказанные значения (train)')
    axes[0, 0].set_ylabel('Остатки')
    axes[0, 0].set_title('Остатки на train')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Остатки vs Предсказания (test)
    axes[0, 1].scatter(test_pred, test_residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Предсказанные значения (test)')
    axes[0, 1].set_ylabel('Остатки')
    axes[0, 1].set_title('Остатки на test')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Гистограмма остатков (train)
    axes[1, 0].hist(train_residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Остатки')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение остатков (train)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    from scipy import stats
    stats.probplot(train_residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q plot остатков')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Анализ остатков - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'reports/learning_curves/{model_name}_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Сохранено")

# ========== 4. ВАЖНОСТЬ ПРИЗНАКОВ ==========
def plot_feature_importance(model, model_name, feature_names):
    """Важность признаков (коэффициенты)"""
    print(f"Строю feature importance для {model_name}...")
    
    if not hasattr(model, 'coef_'):
        print(f"   Модель {model_name} не имеет коэффициентов")
        return
    
    coef = model.coef_
    importance = np.abs(coef)
    
    # Сортируем
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    sorted_coef = coef[indices]
    
    # График
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Абсолютная важность
    bars1 = axes[0].barh(range(len(sorted_importance)), sorted_importance, color='steelblue')
    axes[0].set_yticks(range(len(sorted_importance)))
    axes[0].set_yticklabels(sorted_features)
    axes[0].set_xlabel('Абсолютная важность |коэффициент|')
    axes[0].set_title('Важность признаков (по модулю)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Значения коэффициентов (со знаком)
    colors = ['green' if c > 0 else 'red' for c in sorted_coef]
    bars2 = axes[1].barh(range(len(sorted_coef)), sorted_coef, color=colors)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_yticks(range(len(sorted_coef)))
    axes[1].set_yticklabels(sorted_features)
    axes[1].set_xlabel('Значение коэффициента')
    axes[1].set_title('Коэффициенты модели (со знаком)')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Feature importance - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'reports/learning_curves/{model_name}_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сохраняем данные
    data = {
        'features': sorted_features,
        'coefficients': sorted_coef.tolist(),
        'importance': sorted_importance.tolist()
    }
    with open(f'metrics/learning_curves/{model_name}_importance.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Сохранено")

# ========== ОСНОВНОЙ ЦИКЛ ==========
def get_model_params(model_dict):
    """Убирает служебные параметры ('enabled') из словаря"""
    return {k: v for k, v in model_dict.items() if k != 'enabled'}

models = {
    'ridge': Ridge(**get_model_params(model_params['ridge'])),
    'lasso': Lasso(**get_model_params(model_params['lasso'])),
    'elastic': ElasticNet(**get_model_params(model_params['elastic']))
}

for model_name, model in models.items():
    if not model_params[model_name]['enabled']:
        continue
        
    print(f"\n{'='*50}")
    print(f"Анализ для {model_name}")
    print(f"{'='*50}")
    
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # 1. Кривая обучения
    plot_learning_curve(model, model_name, X_train, y_train)
    
    # 2. Кривая сходимости
    plot_convergence_curve(model, model_name, X_train, y_train)
    
    # 3. Анализ остатков
    plot_residuals(model, model_name, X_train, y_train, X_test, y_test)
    
    # 4. Важность признаков
    plot_feature_importance(model, model_name, X_train.columns.tolist())

print("\nВсе кривые обучения построены!")
print(f"Графики сохранены в: reports/learning_curves/")
print(f"Данные сохранены в: metrics/learning_curves/")