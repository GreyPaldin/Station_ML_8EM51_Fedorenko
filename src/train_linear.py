import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    median_absolute_error
)

# ============================================
# ЗАГРУЗКА ПАРАМЕТРОВ
# ============================================
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
    model_params = params['linear_models']

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================
print("📥 Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()

X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ============================================
# ФУНКЦИЯ ДЛЯ РАСЧЕТА ВСЕХ МЕТРИК
# ============================================
def calculate_all_metrics(y_true, y_pred, y_train=None):
    """
    Рассчитывает ВСЕ возможные метрики регрессии
    
    Args:
        y_true: реальные значения
        y_pred: предсказанные значения
        y_train: обучающие значения (для некоторых метрик)
    
    Returns:
        dict: словарь со всеми метриками
    """
    metrics = {}
    
    # 1. Метрики ошибки
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['median_ae'] = float(median_absolute_error(y_true, y_pred))
    metrics['max_error'] = float(max_error(y_true, y_pred))
    
    # 2. Процентные метрики
    try:
        metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
    except:
        metrics['mape'] = None  # если есть нули в y_true
    
    # 3. Метрики качества
    metrics['r2'] = float(r2_score(y_true, y_pred))
    metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
    
    # 4. Дополнительные статистики
    residuals = y_true - y_pred
    metrics['residuals_mean'] = float(np.mean(residuals))
    metrics['residuals_std'] = float(np.std(residuals))
    metrics['residuals_skew'] = float(pd.Series(residuals).skew())
    
    # 5. Относительные метрики (если есть y_train)
    if y_train is not None:
        y_mean = np.mean(y_train)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        metrics['r2_adj'] = float(1 - (1 - metrics['r2']) * (len(y_true) - 1) / (len(y_true) - X_train.shape[1] - 1))
    
    return metrics

# ============================================
# ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ МОДЕЛИ И МЕТРИК
# ============================================
def train_and_save_model(model, model_name, params_used):
    """
    Обучает модель, считает метрики и сохраняет результат
    """
    print(f"\n{'='*50}")
    print(f"🚀 Обучение: {model_name}")
    print(f"{'='*50}")
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Метрики
    metrics = {
        'model_name': model_name,
        'params': params_used,
        'train': calculate_all_metrics(y_train, y_pred_train),
        'val': calculate_all_metrics(y_val, y_pred_val, y_train),
        'test': calculate_all_metrics(y_test, y_pred_test, y_train),
        'coefficients': {
            'intercept': float(model.intercept_) if hasattr(model, 'intercept_') and model.intercept_ is not None else 0,
            'coef': model.coef_.tolist() if hasattr(model, 'coef_') else []
        },
        'feature_names': X_train.columns.tolist(),
        'n_features': X_train.shape[1],
        'n_samples': {
            'train': len(y_train),
            'val': len(y_val),
            'test': len(y_test)
        }
    }
    
    # Вывод результатов
    print(f"\n📊 Метрики на валидации:")
    print(f"   RMSE: {metrics['val']['rmse']:.4f}")
    print(f"   R²:   {metrics['val']['r2']:.4f}")
    print(f"   MAE:  {metrics['val']['mae']:.4f}")
    
    print(f"\n📊 Метрики на тесте:")
    print(f"   RMSE: {metrics['test']['rmse']:.4f}")
    print(f"   R²:   {metrics['test']['r2']:.4f}")
    print(f"   MAE:  {metrics['test']['mae']:.4f}")
    
    # Сохранение
    os.makedirs('models/linear', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    model_path = f'models/linear/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Модель сохранена: {model_path}")
    
    metrics_path = f'metrics/{model_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Метрики сохранены: {metrics_path}")
    
    return metrics

# ============================================
# ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ
# ============================================
results = {}

# ---------- RIDGE ----------
if model_params['ridge']['enabled']:
    ridge_params = model_params['ridge'].copy()
    ridge_params.pop('enabled')
    
    model = Ridge(**ridge_params)
    results['ridge'] = train_and_save_model(
        model, 
        'ridge', 
        ridge_params
    )

# ---------- LASSO ----------
if model_params['lasso']['enabled']:
    lasso_params = model_params['lasso'].copy()
    lasso_params.pop('enabled')
    
    model = Lasso(**lasso_params)
    results['lasso'] = train_and_save_model(
        model, 
        'lasso', 
        lasso_params
    )

# ---------- ELASTIC NET ----------
if model_params['elastic']['enabled']:
    elastic_params = model_params['elastic'].copy()
    elastic_params.pop('enabled')
    
    model = ElasticNet(**elastic_params)
    results['elastic'] = train_and_save_model(
        model, 
        'elastic', 
        elastic_params
    )

# ============================================
# СОХРАНЕНИЕ СВОДНЫХ МЕТРИК
# ============================================
if results:
    summary = {
        'models_trained': list(results.keys()),
        'best_by_rmse': min(results.keys(), key=lambda x: results[x]['val']['rmse']),
        'best_by_r2': max(results.keys(), key=lambda x: results[x]['val']['r2']),
        'results': {
            name: {
                'val_rmse': res['val']['rmse'],
                'val_r2': res['val']['r2'],
                'test_rmse': res['test']['rmse'],
                'test_r2': res['test']['r2']
            } for name, res in results.items()
        }
    }
    
    with open('metrics/linear_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("📊 СВОДКА ПО МОДЕЛЯМ")
    print("="*50)
    print(f"Лучшая по RMSE: {summary['best_by_rmse']}")
    print(f"Лучшая по R²:   {summary['best_by_r2']}")
    print("\nДетали:")
    for name, res in summary['results'].items():
        print(f"\n{name.upper()}:")
        print(f"  Val RMSE: {res['val_rmse']:.4f}, R²: {res['val_r2']:.4f}")
        print(f"  Test RMSE: {res['test_rmse']:.4f}, R²: {res['test_r2']:.4f}")

print("\n✅ Обучение всех линейных моделей завершено!")