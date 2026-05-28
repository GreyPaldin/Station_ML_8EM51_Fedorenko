import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
from sklearn.model_selection import learning_curve
import yaml

# ========== ЗАГРУЗКА ПАРАМЕТРОВ ==========
with open('params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
    cb_params = params['catboost_models']

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

os.makedirs('models/catboost', exist_ok=True)
os.makedirs('metrics/catboost', exist_ok=True)
os.makedirs('reports/catboost/learning_curves', exist_ok=True)
os.makedirs('reports/catboost/importance', exist_ok=True)

# ========== КОНВЕРТЕР ДЛЯ JSON ==========
def convert_to_serializable(obj):
    """Рекурсивно конвертирует numpy типы в стандартные Python типы"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ========== ФУНКЦИЯ ДЛЯ ВСЕХ МЕТРИК ==========
def calculate_all_metrics(y_true, y_pred, y_train=None):
    metrics = {}
    
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['median_ae'] = float(median_absolute_error(y_true, y_pred))
    metrics['max_error'] = float(max_error(y_true, y_pred))
    
    try:
        metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
    except:
        metrics['mape'] = None
    
    metrics['r2'] = float(r2_score(y_true, y_pred))
    metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
    
    residuals = y_true - y_pred
    metrics['residuals_mean'] = float(np.mean(residuals))
    metrics['residuals_std'] = float(np.std(residuals))
    metrics['residuals_skew'] = float(pd.Series(residuals).skew())
    
    return metrics

# ========== КРИВАЯ ОБУЧЕНИЯ ==========
def plot_learning_curve(model, model_name, X, y):
    print(f"Learning curve для {model_name}...")
    
    train_sizes = np.linspace(0.1, 1.0, 8) * len(X)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        size = int(size)
        X_subset = X[:size]
        y_subset = y[:size]
        
        # Создаем Pool для CatBoost
        train_pool = Pool(X_subset, y_subset)
        val_pool = Pool(X_val, y_val)
        
        # Копируем модель с теми же параметрами
        params_copy = model.get_params()
        params_copy['verbose'] = False
        model_copy = CatBoostRegressor(**params_copy)
        
        model_copy.fit(train_pool, eval_set=val_pool, verbose=False, early_stopping_rounds=10)
        
        train_pred = model_copy.predict(X_subset)
        val_pred = model_copy.predict(X_val)
        
        train_scores.append(mean_squared_error(y_subset, train_pred))
        val_scores.append(mean_squared_error(y_val, val_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Train MSE', color='blue', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation MSE', color='red', linewidth=2)
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('MSE')
    plt.title(f'Кривая обучения - CatBoost {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reports/catboost/learning_curves/{model_name}_learning.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'train_sizes': train_sizes.tolist(),
        'train_mse': train_scores,
        'val_mse': val_scores
    }

# ========== FEATURE IMPORTANCE ==========
def plot_feature_importance(model, model_name, feature_names):
    print(f"Feature importance для {model_name}...")
    
    # Получаем важность признаков
    importance = model.get_feature_importance()
    indices = np.argsort(importance)[::-1]
    
    # График важности
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(importance)), importance[indices], color='steelblue')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Признаки')
    plt.ylabel('Важность')
    plt.title(f'Feature Importance - CatBoost {model_name}')
    
    # Добавляем значения на столбцы
    for bar, val in zip(bars, importance[indices]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'reports/catboost/importance/{model_name}_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сохраняем данные важности
    importance_data = {
        'features': [feature_names[i] for i in indices],
        'importance': importance[indices].tolist()
    }
    
    with open(f'metrics/catboost/{model_name}_importance.json', 'w') as f:
        json.dump(importance_data, f, indent=2)
    
    return importance_data

# ========== ОБУЧЕНИЕ МОДЕЛИ ==========
def train_and_save_model(model_config, model_name):
    print(f"\n{'='*50}")
    print(f"Обучение CatBoost: {model_name}")
    print(f"{'='*50}")
    
    # Убираем 'enabled' из параметров
    params = {k: v for k, v in model_config.items() if k != 'enabled'}
    
    # Создаем модель
    model = CatBoostRegressor(**params)
    
    # Создаем Pool для валидации
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    
    # Обучаем с валидацией
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=False,
        plot=False,
        early_stopping_rounds=params.get('early_stopping_rounds', 50)
    )
    
    # Получаем лучшую итерацию
    best_iteration = model.get_best_iteration()
    print(f"   Лучшая итерация: {best_iteration}")
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Метрики
    metrics = {
        'model_name': model_name,
        'model_type': 'CatBoost',
        'params': convert_to_serializable(params),
        'best_iteration': int(best_iteration) if best_iteration else None,
        'train': calculate_all_metrics(y_train, y_pred_train),
        'val': calculate_all_metrics(y_val, y_pred_val, y_train),
        'test': calculate_all_metrics(y_test, y_pred_test, y_train),
        'n_features': int(X_train.shape[1]),
        'n_samples': {
            'train': int(len(y_train)),
            'val': int(len(y_val)),
            'test': int(len(y_test))
        }
    }
    
    # Вывод
    print(f"\nМетрики на тесте:")
    print(f"   R²:   {metrics['test']['r2']:.4f}")
    print(f"   RMSE: {metrics['test']['rmse']:.4f}")
    print(f"   MAE:  {metrics['test']['mae']:.4f}")
    
    # Кривая обучения
    learning_data = plot_learning_curve(model, model_name, X_train, y_train)
    metrics['learning_curve'] = learning_data
    
    # Feature importance
    importance_data = plot_feature_importance(model, model_name, X_train.columns.tolist())
    metrics['feature_importance'] = importance_data
    
    # Сохранение
    model_path = f'models/catboost/{model_name}.cbm'
    model.save_model(model_path)
    print(f"\nМодель сохранена: {model_path}")
    
    metrics = convert_to_serializable(metrics)
    metrics_path = f'metrics/catboost/{model_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Метрики сохранены: {metrics_path}")
    
    return model, metrics

# ========== ОСНОВНОЙ ЦИКЛ ==========
results = {}
trained_models = {}

for model_name, model_config in cb_params.items():
    if not model_config.get('enabled', True):
        continue
    
    trained_models[model_name], results[model_name] = train_and_save_model(
        model_config, model_name
    )

# ========== СВОДКА ==========
if results:
    summary = {
        'models_trained': list(results.keys()),
        'best_by_r2': max(results.keys(), key=lambda x: results[x]['test']['r2']),
        'best_by_rmse': min(results.keys(), key=lambda x: results[x]['test']['rmse']),
        'results': {
            name: {
                'test_r2': res['test']['r2'],
                'test_rmse': res['test']['rmse'],
                'test_mae': res['test']['mae'],
                'best_iteration': res.get('best_iteration')
            } for name, res in results.items()
        }
    }
    
    with open('metrics/catboost/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("СВОДКА ПО МОДЕЛЯМ CATBOOST")
    print("="*50)
    print(f"Лучшая по R²:   {summary['best_by_r2']}")
    print(f"Лучшая по RMSE: {summary['best_by_rmse']}")
    
    for name, res in summary['results'].items():
        print(f"\n{name.upper()}:")
        print(f"  Test R²: {res['test_r2']:.4f}, RMSE: {res['test_rmse']:.4f}")
        if res.get('best_iteration'):
            print(f"  Best iter: {res['best_iteration']}")

print("\nОбучение CatBoost завершено!")