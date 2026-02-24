import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import yaml

# ========== ЗАГРУЗКА ПАРАМЕТРОВ ==========
with open('params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
    model_params = params['linear_models']

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("📥 Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

os.makedirs('reports/learning_curves', exist_ok=True)
os.makedirs('metrics/learning_curves', exist_ok=True)

# ========== 1. КРИВАЯ ОБУЧЕНИЯ ==========
def plot_learning_curve(model, model_name, X, y):
    """Кривая обучения: ошибка от размера выборки"""
    print(f"📈 Learning curve для {model_name}...")
    
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
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Train MSE', color='blue', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation MSE', color='red', linewidth=2)
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('MSE')
    plt.title(f'Кривая обучения - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reports/learning_curves/{model_name}_learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    data = {
        'train_sizes': train_sizes.tolist(),
        'train_mse': train_scores,
        'val_mse': val_scores
    }
    with open(f'metrics/learning_curves/{model_name}_learning.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   ✅ Сохранено")

# ========== 2. АНАЛИЗ ОСТАТКОВ ==========
def plot_residuals(model, model_name, X_train, y_train, X_test, y_test):
    """Анализ остатков: предсказания vs ошибки"""
    print(f"📊 Residuals plot для {model_name}...")
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_residuals = y_train - train_pred
    test_residuals = y_test - test_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Остатки vs Предсказания (train)
    axes[0, 0].scatter(train_pred, train_residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Предсказанные значения (train)')
    axes[0, 0].set_ylabel('Остатки')
    axes[0, 0].set_title('Остатки на train')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Остатки vs Предсказания (test)
    axes[0, 1].scatter(test_pred, test_residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Предсказанные значения (test)')
    axes[0, 1].set_ylabel('Остатки')
    axes[0, 1].set_title('Остатки на test')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Гистограмма остатков (train)
    axes[1, 0].hist(train_residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Остатки')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение остатков (train)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(train_residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q plot остатков')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Анализ остатков - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'reports/learning_curves/{model_name}_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено")

# ========== 3. ВАЖНОСТЬ ПРИЗНАКОВ ==========
def plot_feature_importance(model, model_name, feature_names):
    """Важность признаков (коэффициенты)"""
    print(f"📊 Feature importance для {model_name}...")
    
    if not hasattr(model, 'coef_'):
        print(f"   ⚠️ Нет коэффициентов")
        return
    
    coef = model.coef_
    importance = np.abs(coef)
    
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    sorted_coef = coef[indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Абсолютная важность
    axes[0].barh(range(len(sorted_importance)), sorted_importance, color='steelblue')
    axes[0].set_yticks(range(len(sorted_importance)))
    axes[0].set_yticklabels(sorted_features)
    axes[0].set_xlabel('Абсолютная важность |коэффициент|')
    axes[0].set_title('Важность признаков (по модулю)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Значения коэффициентов
    colors = ['green' if c > 0 else 'red' for c in sorted_coef]
    axes[1].barh(range(len(sorted_coef)), sorted_coef, color=colors)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_yticks(range(len(sorted_coef)))
    axes[1].set_yticklabels(sorted_features)
    axes[1].set_xlabel('Значение коэффициента')
    axes[1].set_title('Коэффициенты модели')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Feature importance - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'reports/learning_curves/{model_name}_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    data = {
        'features': sorted_features,
        'coefficients': sorted_coef.tolist(),
        'importance': sorted_importance.tolist()
    }
    with open(f'metrics/learning_curves/{model_name}_importance.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   ✅ Сохранено")

# ========== ОСНОВНОЙ ЦИКЛ ==========
def get_model_params(model_dict):
    """Убирает 'enabled' из параметров"""
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
    print(f"📊 Анализ для {model_name}")
    print(f"{'='*50}")
    
    model.fit(X_train, y_train)
    plot_learning_curve(model, model_name, X_train, y_train)
    plot_residuals(model, model_name, X_train, y_train, X_test, y_test)
    plot_feature_importance(model, model_name, X_train.columns.tolist())

print("\n✅ Все кривые обучения построены!")