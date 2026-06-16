import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml
import time
import sys
improt tensorboard

# ========== ЗАГРУЗКА ПАРАМЕТРОВ ==========
print("Загрузка параметров...")
with open('params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
    nn_params = params['neural_networks']

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("Нормализация данных...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

os.makedirs('models/neural', exist_ok=True)
os.makedirs('metrics/neural', exist_ok=True)
os.makedirs('reports/neural/learning_curves', exist_ok=True)
os.makedirs('reports/neural/weights', exist_ok=True)

def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def calculate_all_metrics(y_true, y_pred):
    metrics = {}
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['median_ae'] = float(median_absolute_error(y_true, y_pred))
    metrics['max_error'] = float(max_error(y_true, y_pred))
    metrics['r2'] = float(r2_score(y_true, y_pred))
    residuals = y_true - y_pred
    metrics['residuals_mean'] = float(np.mean(residuals))
    metrics['residuals_std'] = float(np.std(residuals))
    metrics['residuals_skew'] = float(pd.Series(residuals).skew())
    return metrics

def plot_learning_curve(model, model_name, X, y, X_val, y_val):
    train_sizes = np.linspace(0.1, 1.0, 6) * len(X)
    train_mse, val_mse = [], []
    for size in train_sizes:
        size = int(size)
        Xs = X[:size]
        ys = y[:size]
        m = MLPRegressor(**model.get_params())
        m.max_iter = 100
        m.fit(Xs, ys)
        train_mse.append(mean_squared_error(ys, m.predict(Xs)))
        val_mse.append(mean_squared_error(y_val, m.predict(X_val)))
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mse, 'o-', label='Train MSE', color='blue')
    plt.plot(train_sizes, val_mse, 'o-', label='Validation MSE', color='red')
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('MSE')
    plt.title(f'Кривая обучения - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'reports/neural/learning_curves/{model_name}_learning.png', dpi=150, bbox_inches='tight')
    plt.close()
    return {'train_sizes': train_sizes.tolist(), 'train_mse': train_mse, 'val_mse': val_mse}

def plot_loss_curve(model, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Train loss', color='blue')
    plt.xlabel('Итерации')
    plt.ylabel('Loss')
    plt.title(f'Кривая потерь - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'reports/neural/learning_curves/{model_name}_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    return [float(x) for x in model.loss_curve_]

def plot_weight_histograms(model, model_name):
    n_layers = len(model.coefs_)
    fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    weight_stats = {}
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        axes[0, i].hist(coef.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, i].set_title(f'Слой {i+1} - Веса')
        axes[0, i].axvline(x=0, color='red', linestyle='--')
        axes[1, i].hist(intercept, bins=30, alpha=0.7, color='coral', edgecolor='black')
        axes[1, i].set_title(f'Слой {i+1} - Смещения')
        axes[1, i].axvline(x=0, color='red', linestyle='--')
        weight_stats[f'layer_{i+1}'] = {
            'weights_mean': float(np.mean(coef)),
            'weights_std': float(np.std(coef)),
            'bias_mean': float(np.mean(intercept))
        }
    plt.suptitle(f'Распределение весов - {model_name}')
    plt.tight_layout()
    plt.savefig(f'reports/neural/weights/{model_name}_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    return weight_stats

def plot_feature_importance(model, model_name, feature_names):
    """График важности признаков для нейросети"""
    
    # СОЗДАЁМ ПАПКУ ЕСЛИ НЕТ
    os.makedirs('reports/neural/importance', exist_ok=True)
    
    # Берём веса первого слоя
    weights_layer1 = model.coefs_[0]
    importance = np.mean(np.abs(weights_layer1), axis=1)
    
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(importance)), importance[indices], color='steelblue')
    plt.yticks(range(len(importance)), [feature_names[i] for i in indices])
    plt.xlabel('Средняя абсолютная важность')
    plt.title(f'Важность признаков (входной слой) - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'reports/neural/importance/{model_name}_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}

# ========== ОСНОВНОЙ ЦИКЛ ==========
results = {}

for model_name, model_cfg in nn_params.items():
    if not model_cfg.get('enabled', True):
        continue
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ: {model_name}")
    print(f"{'='*60}")

    # Обработка параметров (преобразование tol в число)
    params = {}
    for k, v in model_cfg.items():
        if k == 'enabled':
            continue
        if k == 'tol' and isinstance(v, str):
            v = float(v)
        params[k] = v
    
    print(f"Параметры: {params}")

    model = MLPRegressor(**params)


    start = time.time()
    model.fit(X_train_scaled, y_train_scaled)
    train_time = time.time() - start

    # Предсказания и обратное масштабирование
    y_pred_train = scaler_y.inverse_transform(model.predict(X_train_scaled).reshape(-1, 1)).ravel()
    y_pred_val   = scaler_y.inverse_transform(model.predict(X_val_scaled).reshape(-1, 1)).ravel()
    y_pred_test  = scaler_y.inverse_transform(model.predict(X_test_scaled).reshape(-1, 1)).ravel()

    metrics = {
        'model_name': model_name,
        'params': params,
        'train': calculate_all_metrics(y_train, y_pred_train),
        'val':   calculate_all_metrics(y_val,   y_pred_val),
        'test':  calculate_all_metrics(y_test,  y_pred_test),
        'n_iter': int(model.n_iter_),
        'training_time_sec': train_time,
        'n_features': int(X_train.shape[1]),
        'n_samples': {'train': len(y_train), 'val': len(y_val), 'test': len(y_test)},
        'learning_curve': plot_learning_curve(model, model_name, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled),
        'loss_curve': plot_loss_curve(model, model_name) if hasattr(model, 'loss_curve_') else [],
        'weight_stats': plot_weight_histograms(model, model_name)
    }
    
    # Сохранение
    joblib.dump(model, f'models/neural/{model_name}.pkl')
    joblib.dump({'X': scaler_X, 'y': scaler_y}, f'models/neural/{model_name}_scaler.pkl')
    with open(f'metrics/neural/{model_name}_metrics.json', 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)

    print(f"{model_name}: R² = {metrics['test']['r2']:.4f}, RMSE = {metrics['test']['rmse']:.4f}, итераций = {model.n_iter_}")

    results[model_name] = metrics
    feature_names = X_train.columns.tolist()
    importance_dict = plot_feature_importance(model, model_name, feature_names)
    print(f"   Важность признаков: {importance_dict}")
# ========== СВОДКА ==========
if results:
    print("\n" + "="*60)
    print("СВОДКА ПО НЕЙРОСЕТЯМ")
    print("="*60)
    best_r2 = max(results.keys(), key=lambda x: results[x]['test']['r2'])
    best_rmse = min(results.keys(), key=lambda x: results[x]['test']['rmse'])
    print(f"Лучшая по R²:   {best_r2} (R² = {results[best_r2]['test']['r2']:.4f})")
    print(f"Лучшая по RMSE: {best_rmse} (RMSE = {results[best_rmse]['test']['rmse']:.4f})")
    print(f"Общее время: {sum(m['training_time_sec'] for m in results.values()):.1f} сек")
    for name, m in results.items():
        print(f"\n{name.upper()}: R²={m['test']['r2']:.4f}, RMSE={m['test']['rmse']:.4f}, iter={m['n_iter']}")

print("\nОБУЧЕНИЕ НЕЙРОСЕТЕЙ ЗАВЕРШЕНО!")